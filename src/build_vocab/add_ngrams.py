import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.constants import *
from collections import defaultdict
import utils.utils_funcs as utils
from math import log
from transformers import AutoTokenizer
import logging
import pickle
import pickle
import pickle
import pickle
import copy
from datasets import Dataset

import build_vocab.handle_ds_pt as handle_ds_pt
from utils.constants import *

# ----- main function: --------
def find_ngrams_to_add(tokenized_target_ds:Dataset,cfg:Dict)->Dict[str,List[str]]:
    
    flatten_text = [token for sentence in tokenized_target_ds for token in sentence]
    ngrams_freqs = get_ngram_freqs(flatten_text, cfg['ngram_k'],cfg)

    # trim ngrams for faster run, not all ngrams are candidates for adding or affecting
    #ngrams_freqs = trim_ngrams_by_freq(ngrams_freqs,len(flatten_text))
    ngrams_freqs = remove_numbers_ngrams(ngrams_freqs)
    ngrams_freqs = remove_problematic_ngrams(ngrams_freqs)
    ngrams_scores = calculate_ngrams_scores(ngrams_freqs)
    ngrams_list = sorted(ngrams_scores.items(), key=lambda x: x[1], reverse=True)
    
    # TODO: take into account affected dict - NOT IMPLEMENTED YET
    ngrams_to_add = ngrams_list[:cfg['num_to_add']]
    added_ngrams={ngram: scr for ngram, scr in ngrams_to_add}
    return added_ngrams

def remove_numbers_ngrams(ngrams_freqs:Dict[str,int])->Dict[str,int]:
    # remove ngrams that contain only numbers
    ngrams_freqs_no_numbers = {ngram: freq for ngram, freq in ngrams_freqs.items() if not all(token.isdigit() for token in ngram)}
    utils.log_w_wandb(logger=logger,message=f"Removed ngrams that contain only numbers, original ngrams freq size: {len(ngrams_freqs)}, trimmed ngrams freq size: {len(ngrams_freqs_no_numbers)}")
    return ngrams_freqs_no_numbers

def remove_problematic_ngrams(ngrams_freqs:Dict[str,int])->Dict[str,int]:
    # don't add <0xE2><0x82><0xA9> ngrams
    # to solve this: self.decode(converted_token_ids)
    #'According to the South Korean National Tax Service, the average annual earnings for a Korean idol in 2013 were KR<0xE2><0x82><0xA9>46.74 million (US$42,000). This was almost double the 2010 figure of KR<0xE2><0x82><0xA9>26.97 million (US$25,275), a rise attributable to the global spread of "Hallyu" in recent years.'
    #texts
    #'According to the South Korean National Tax Service, the average annual earnings for a Korean idol in 2013 were KR₩46.74 million (US$42,000). This was almost double the 2010 figure of KR₩26.97 million (US$25,275), a rise attributable to the global spread of "Hallyu" in recent years.'
    ngrams_freqs_wout_problematics = defaultdict(int)
    for ngram,freq in ngrams_freqs.items():
        prob_ngram=False
        for token in ngram:
            if '<0x' in token:
                print(f"problematic ngram: {ngram}")
                prob_ngram=True
                break
        if not prob_ngram:
            ngrams_freqs_wout_problematics[ngram] = freq
    utils.log_w_wandb(logger=logger,message=f"Removed ngrams that contain '<0x' kind of things, original ngrams freq size: {len(ngrams_freqs)}, new ngrams freq size: {len(ngrams_freqs_wout_problematics)}")
    return ngrams_freqs_wout_problematics
    
# TODO: currently count ngrams crossing sentences:
def get_ngram_freqs(flatten_text:List[str], k:int,cfg:Dict):
    
    def calculate_ngram_freqs(flatten_text:List[str], k:int,use_ngrams:bool=False):
        space_token='▁'
        ngram_freqs = defaultdict(int)
        
        for i in range(len(flatten_text)):
            for n in range(2, k+1):
                if i + n <= len(flatten_text):
                    ngram = tuple(flatten_text[i:i+n])
                    is_ngram=False
                    for i, token in enumerate(ngram):
                        # if the token ends with a space and not the last in the ngram, dont count it:
                        if token.endswith(space_token) and i != len(ngram)-1:
                            is_ngram=True
                            print(f'Not adding ngram: {ngram}')
                            break
                    if (not is_ngram) or (is_ngram and use_ngrams):
                        ngram_freqs[ngram] += 1
        return ngram_freqs

    path = os.path.join(DATASET_STATS_PATH ,utils.cfg_to_filename(cfg)+'_ngram_freqs.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as file:
            ngram_freqs = pickle.load(file)
        utils.log_w_wandb(logger=logger,message=f"ngrams freqs loaded from {path}")
    else:
        ngram_freqs = calculate_ngram_freqs(flatten_text=flatten_text,k=k)
        utils.log_w_wandb(logger=logger,message=f"ngrams freqs saved to {path}")
        with open(path, 'wb') as file:
            pickle.dump(ngram_freqs, file)
    return ngram_freqs

def trim_ngrams_by_freq(ngrams_freq_dict:Dict[str,int])->Dict[str,int]:
    # Calculate the total number of ngrams to keep
    trimed_ngrams_dict=defaultdict(int)
    for ngram, score in ngrams_freq_dict.items():
        if score > FREQ_TRIM_THRESHOLD:
            trimed_ngrams_dict[ngram] = score
    utils.log_w_wandb(logger=logger,message=f"Trimed ngrams by freq, original ngrams freq size: {len(ngrams_freq_dict)}, trimmed ngrams freq size: {len(trimed_ngrams_dict)}")

    return trimed_ngrams_dict
    

def calculate_ngrams_probs(ngram_freqs_sorted):
    # without log
    ngrams_prob=defaultdict(int)
    sum_of_appears=sum(freq for _, freq in ngram_freqs_sorted)
    for ngram,ngram_freq in ngram_freqs_sorted:
        c_ngram_prob = ngram_freq/sum_of_appears
        ngrams_prob[ngram] = c_ngram_prob
    return ngrams_prob

def calculate_ngrams_scores(ngrams_freqs:Dict[str,int]):
    ngram_scores=defaultdict(int)
    for ngram, freq in ngrams_freqs.items():
        ngram_cont=(len(ngram)-1)
        ngram_scores[ngram] = freq*ngram_cont
    return ngram_scores
   
   
# ----------------- affected dict: -----------------

def get_affected_ngrams_dict(tokenized_text, max_n, ngrams_scores_dict, cfg):
    path_to_dir = utils.adjust_path_to_server(f'{os.path.expanduser("~")}/tokens_count/data/runs_stats/ngram_stats_add_ngram')
    num_of_ngrams=len(ngrams_scores_dict)
    file_name = utils.cfg_to_name_add_ngrams(cfg=cfg, num_of_ngrams=num_of_ngrams,file_type=None)
    file_name+='_affected_ngrams.pkl'
    path = os.path.join(path_to_dir, file_name)
    
    if os.path.exists(path):
        with open(path, 'rb') as file:
            affected_ngrams_dict = pickle.load(file)
        utils.log_w_wandb(logger=logger, message=f"Affected ngrams dict loaded from {path}")
    else:
        affected_ngrams_dict = {}
        affected_ngrams_dict = find_affected_n_grams(tokenized_text=tokenized_text, max_n=max_n, ngrams_scores_dict=ngrams_scores_dict, affected_n_grams={})
        with open(path, 'wb') as file:
            pickle.dump(affected_ngrams_dict, file)
    
    return affected_ngrams_dict

def find_affected_n_grams(tokenized_text, max_n,
                          ngrams_scores_dict,affected_n_grams):
    print(f"started affected ngrams")
    tokenized_text = tokenized_text
    def add_to_affected_n_grams(n_gram1, n_gram2): #ngram1 is the dominate ngram
        if n_gram1 not in affected_n_grams:
            affected_n_grams[n_gram1] = {n_gram2: 1}
        else:
            if n_gram2 not in affected_n_grams[n_gram1]:
                affected_n_grams[n_gram1][n_gram2] = 1
            else:
                affected_n_grams[n_gram1][n_gram2] += 1
    min_n = 2
    cntr = 0
    flatten_text = [token for sentence in tokenized_text for token in sentence]
    tokens_in_text = len(flatten_text)
    for sentence in tokenized_text:
        for start1 in range(len(sentence)):
            cntr += 1
            if cntr % 400000 == 0:
                logger.log(logging.INFO, f"finished {cntr} out of {tokens_in_text} tokens")
            for end1 in range(start1+(min_n),(start1)+(max_n)+1):
                for start2 in range(start1+1, end1):
                    for end2 in range(start2+(min_n),(start2)+(max_n)+1):
                        if end2 >= len(sentence) or start2>end1:
                            continue
                        ngram1 = tuple(sentence[start1:end1])
                        ngram2 = tuple(sentence[start2:end2])
                        if ngram1 not in ngrams_scores_dict or ngram2 not in ngrams_scores_dict:
                            continue
                        add_to_affected_n_grams(ngram1, ngram2)
                        add_to_affected_n_grams(ngram2, ngram1)
    return affected_n_grams

def check_if_crossing(start_1,end_1,start_2):
    ngram1_subof_ngram2=(start_1 < start_2 and end_1 >= start_2) #  (a,b,c) (b,c) 'abcd' or (a,b,c,d) (b,c) 'abcd'
    ngram2_after_ngram1=(end_1 == start_2)
    #ngram2_subof_ngram1=(start_2 < start_1 and end_2 >= start_1) cant be, start_2 > start_1 always
    # ngram1_after_ngram2=(end_2 == start_1 and ngram2[-1] == ngram1[0]) cant be, start_2 > start_1 always
    return ngram1_subof_ngram2 or ngram2_after_ngram1
              
def add_ngram_to_vocab(ngrams_scores_sorted, affected_dict,ngram_freqs_sorted_dict,ngrams_prob,target_corpus_tokens_probs,ngram_num):
    dominate_ngram, dominate_ngram_score = ngrams_scores_sorted.pop(ngram_num)
    assert dominate_ngram in affected_dict 
    new_affected_dict = copy.deepcopy(affected_dict)
    for affected_ngram, count in affected_dict[dominate_ngram].items():
        # Reduce the frequency of affected ngram based on the overlap count
        ngram_freqs_sorted_dict[affected_ngram] -= count
        # Ensure frequency doesn't go below zero
        if ngram_freqs_sorted_dict[affected_ngram] < 0:
            ngram_freqs_sorted_dict[affected_ngram] = 0
            print(f"ngram {affected_ngram} got below 0, need to check how !!")
        assert ngram_freqs_sorted_dict[affected_ngram] >= 0
        # remove the 'dominated ngram' from the affected ngram dict. its not affected by 'a weaker' ngram
        if dominate_ngram in affected_dict.get(affected_ngram, {}):
            del new_affected_dict[affected_ngram][dominate_ngram]
        else:
            print(f"'{dominate_ngram}' not found in the inner dictionary of '{affected_ngram}, shouldn't happen !'")
        ngrams_prob[affected_ngram] = ngram_freqs_sorted_dict[affected_ngram] / sum(ngram_freqs_sorted_dict.values())
    ngrams_scores_sorted = calculate_ngrams_scores(ngrams_prob, target_corpus_tokens_probs)
    ngrams_scores_sorted = sorted(ngrams_scores_sorted.items(), key=lambda x: x[1], reverse=True)
    return dominate_ngram,ngram_freqs_sorted_dict, ngrams_scores_sorted,new_affected_dict
       
# from add ngrams:

    # affected - currently not implemented:
    #affected_n_grams={}
    #ngrams_scores_sorted = sorted(ngrams_scores_dict.items(), key=lambda x: x[1], reverse=True)
    #ngram_freqs_sorted_dict = dict(ngram_freqs_sorted)
    #added_ngrams = set()
    #ngrams_scores_dict_org = dict(ngrams_scores_sorted)
    #for i in range(cfg['num_to_add']):
    #    added_ngram = ngrams_scores_sorted.pop(0)[0]
    #    added_ngrams.add(added_ngram)
    #    #added_ngram,ngram_freqs_sorted_dict, ngrams_scores_sorted,affected_n_grams =add_ngram_to_vocab(ngrams_scores_sorted, affected_n_grams,ngram_freqs_sorted_dict,ngrams_prob_dict,target_corpus_tokens_probs,i)
    #    #added_ngrams.add(added_ngram)
    #    # delete added ngram from dicts:
    #    #TODO:
    #added_ngrams_with_score = {ngram: ngrams_scores_dict_org[ngram] for ngram in added_ngrams}
    # Save added_ngrams_with_score to a file
