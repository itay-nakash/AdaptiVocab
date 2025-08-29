import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.constants import *
from typing import List, Dict, Any, Tuple, Union, Optional
from collections import defaultdict
from datasets import load_dataset
from transformers import PreTrainedTokenizer, AutoTokenizer
import json
from tqdm import tqdm
from datasets import Dataset
from utils.constants import *
import utils.utils_funcs as utils
import pickle


class SortedNgramDict:
    def __init__(self):
        from sortedcontainers import SortedList
        self.dict = {}
        self.sorted_list = SortedList(key=lambda item: -1 * SortedNgramDict.score_ngram(item))

    @staticmethod
    def score_ngram(item) -> int:
        ngram, freq = item
        return freq * len(ngram)

    def __setitem__(self, key: Any, value: Any):
        if key in self.dict:
            old_value = self.dict[key]
            self.sorted_list.discard((key, old_value))
        self.dict[key] = value
        self.sorted_list.add((key, value))

    def __getitem__(self, key: Any) -> Any:
        return self.dict[key]

    def __delitem__(self, key: Any):
        value = self.dict.pop(key)
        self.sorted_list.discard((key, value))

    def __contains__(self, key: Any) -> bool:
        return key in self.dict

    def __len__(self) -> int:
        return len(self.dict)

    def pop(self, key: Any) -> Any:
        if key in self.dict:
            value = self.dict.pop(key)
            self.sorted_list.discard((key, value))
            return value
        else:
            raise KeyError(f"Key '{key}' not found")


def preproccess_texts(texts: List[str],
                      tokenizer: PreTrainedTokenizer,
                      min_words: int = 40,
                      max_words: int = 100) -> List[List[str]]:
    tokenized_texts = []
    for text in texts:
        n_words = len(text.split())
        if n_words < min_words:
            continue
        if n_words > max_words:
            # divide the text into chunks of max_words
            text_chunks = [text[i:i + max_words] for i in range(0, n_words, max_words)]
        else:
            text_chunks = [text]
        for text_chunk in text_chunks:
            tokenized_text = tokenizer.tokenize(text_chunk)
            tokenized_texts.append(tokenized_text)
    return tokenized_texts


def remove_numbers_ngrams(ngram_freqs: Dict[Tuple, int]) -> Dict[Tuple, int]:
    # remove ngrams that contain only numbers
    ngram_freqs_no_numbers = {ngram: freq for ngram, freq in ngram_freqs.items() if not all(token.isdigit() for token in ngram)}
    # log_w_wandb(logger=logger,message=f"Removed ngrams that contain only numbers, original ngrams freq size: {len(ngram_freqs)}, trimmed ngrams freq size: {len(ngram_freqs_no_numbers)}")
    return ngram_freqs_no_numbers


def remove_problematic_ngrams(ngram_freqs: Dict[Tuple, int]) -> Dict[Tuple, int]:
    # don't add <0xE2><0x82><0xA9> ngrams
    # to solve this: self.decode(converted_token_ids)
    #'According to the South Korean National Tax Service, the average annual earnings for a Korean idol in 2013 were KR<0xE2><0x82><0xA9>46.74 million (US$42,000). This was almost double the 2010 figure of KR<0xE2><0x82><0xA9>26.97 million (US$25,275), a rise attributable to the global spread of "Hallyu" in recent years.'
    #texts
    #'According to the South Korean National Tax Service, the average annual earnings for a Korean idol in 2013 were KR₩46.74 million (US$42,000). This was almost double the 2010 figure of KR₩26.97 million (US$25,275), a rise attributable to the global spread of "Hallyu" in recent years.'
    ngram_freqs_wout_problematics = defaultdict(int)
    for ngram, freq in ngram_freqs.items():
        prob_ngram = False
        for token in ngram:
            if '<0x' in token or '\xa0' in token:
                # print(f"problematic ngram: {ngram}")
                prob_ngram = True
                break
        if not prob_ngram:
            ngram_freqs_wout_problematics[ngram] = freq
    # log_w_wandb(logger=logger,message=f"Removed ngrams that contain '<0x' kind of things, original ngrams freq size: {len(ngram_freqs)}, new ngrams freq size: {len(ngram_freqs_wout_problematics)}")
    return ngram_freqs_wout_problematics


def get_affected_ngrams(tokenized_texts: List[List[str]],
                        max_n: int, vocab_size: int,
                        min_freq_affected: int = 3) -> Tuple[Dict[Tuple, int], Dict[Tuple, Dict[Tuple, int]]]:
    # The function returns filtered ngram_freqs and affected_dict
    # we start by counting ngram frequencies
    ngram_freqs = defaultdict(int)
    space_token='▁'
    use_ngrams:bool=False
    for tokenized_text in tokenized_texts:
        for n in range(1, max_n + 1):
            for i in range(len(tokenized_text) - n + 1):
                ngram = tuple(tokenized_text[i:i + n])
                is_ngram=False
                for i, token in enumerate(ngram):
                    if i>0 and token.startswith(space_token):
                        is_ngram=True
                        break
                if (not is_ngram) or (is_ngram and use_ngrams):
                    ngram_freqs[ngram] += 1
    unigram_freqs = {ngram: freq for ngram, freq in ngram_freqs.items() if len(ngram) == 1}
    ngram_freqs = {ngram: freq for ngram, freq in ngram_freqs.items() if len(ngram) > 1}
    # remove ngrams that contain only numbers or problematic tokens
    ngram_freqs = remove_numbers_ngrams(ngram_freqs)
    ngram_freqs = remove_problematic_ngrams(ngram_freqs)
    # filter ngram_freqs and keep only 5*vocab size ngrams (and all unigrams)
    ngram_freqs = dict(sorted(ngram_freqs.items(), key=lambda x: x[1], reverse=True)[:5 * vocab_size])
    ngram_freqs.update(unigram_freqs)
    # now we find affected ngrams
    affected_dict = defaultdict(lambda: defaultdict(int))
    for tokenized_text in tqdm(tokenized_texts):
        # for each text, get ngrams and their locations
        ngrams_with_indices = []
        for n in range(1, max_n + 1):
            for i in range(len(tokenized_text) - n + 1):
                ngram = tuple(tokenized_text[i:i + n])
                if ngram in ngram_freqs:
                    ngrams_with_indices.append((ngram, i, i + n - 1))
        ngrams_with_indices = sorted(ngrams_with_indices, key=lambda x: (x[1], x[2]))
        # for each pair of ngrams, check if they overlap (affected)
        for i, (ngram1, start1, end1) in enumerate(ngrams_with_indices):
            ngram_len1 = len(ngram1)
            #  tokens (unigrams) do not affect ngrams, however, ngrams can affect tokens
            if ngram_len1 == 1:  # skip unigrams
                continue
            # this is for optimizing the search, notice that we only need to search for ngrams
            # that are within max_n * max_n tokens from the current ngram (otherwise, there is no overlap)
            start_range = max(0, i - max_n * max_n)
            end_range = min(len(ngrams_with_indices), i + max_n * max_n + 1)
            for j in range(start_range, end_range):
                if i != j:
                    ngram2, start2, end2 = ngrams_with_indices[j]
                    if max(start1, start2) <= min(end1, end2):
                        affected_dict[ngram1][ngram2] += 1
    # make affected_dict a regular dict
    affected_dict = {ngram: {ngram_a: freq for ngram_a, freq in ngram_affected.items() if freq >= min_freq_affected}
                     for ngram, ngram_affected in affected_dict.items()}
    return ngram_freqs, {k: dict(v) for k, v in affected_dict.items()}


def find_new_vocab(ngram_freqs: Dict[Tuple, int],
                   affected_dict: Dict[Tuple, Dict[Tuple, int]],
                   current_vocab: List[str],
                   final_vocab_size: int) -> Dict[Union[str, Tuple[str]], int]:
    current_dict = SortedNgramDict()
    queue_dict = SortedNgramDict()
    current_vocab = set([tuple([token]) for token in current_vocab])
    ngram_freqs.update({ngram: 0 for ngram in current_vocab})
    for ngram, freq in ngram_freqs.items():
        if ngram in current_vocab:
            current_dict[ngram] = freq
        else:
            queue_dict[ngram] = freq
    current_size = len(current_dict)
   # if current_size < final_vocab_size:
   #     for ngram, freq in queue_dict.sorted_list[:final_vocab_size - current_size]:
   #         current_dict[ngram] = freq
   #         # pop from queue_dict
   #         queue_dict.pop(ngram)
   # if current_size > final_vocab_size:
   #     for ngram, freq in current_dict.sorted_list[final_vocab_size:]:
   #         # pop from current_dict
   #         queue_dict[ngram] = freq
   #         current_dict.pop(ngram)
    num_to_add = final_vocab_size-current_size
    to_add_dict={}
    for i in range(num_to_add):
        if len(queue_dict) == 0:
            break
        ngram_to_add = queue_dict.sorted_list[0][0]
        ngram_to_add_freq = queue_dict.pop(ngram_to_add)
        assert len(ngram_to_add) > 1, f"Trying to add unigram: {ngram_to_add}"
        for affected_ngram, affected_freq in affected_dict.get(ngram_to_add, {}).items():
            if affected_ngram in queue_dict:
                queue_dict[affected_ngram] -= affected_freq
        to_add_dict[ngram_to_add] = ngram_to_add_freq
 #       print(f"[{i}/{num_to_add}]\t\t"
 #              f"\t\tAdded: {ngram_to_add} - Freq: {ngram_to_add_freq}")

    return to_add_dict


def main(tokenized_target_ds:Dataset,final_vocab_size:int,current_vocab:List[str],cfg:Dict)->Dict[str,List[str]]:
    path_freqs = os.path.join(DATASET_STATS_PATH ,utils.cfg_to_filename(cfg), '_ngram_freqs.json')
    path_affected = os.path.join(DATASET_STATS_PATH ,utils.cfg_to_filename(cfg), '_affected_dict.json')

    if os.path.exists(path_freqs) and False: #never do this - I want new ngrams!
        ngram_freqs = utils.load_pickle(path_freqs)
        affected_dict = utils.load_pickle(path_affected)
        utils.log_w_wandb(logger=logger,message=f"ngrams freqs & affected dict loaded from {utils.cfg_to_filename(cfg)}")
    else:
        print('Calculating ngrams')
        ngram_freqs, affected_dict = get_affected_ngrams(tokenized_target_ds, cfg['ngram_k'], final_vocab_size)
       # ngram_freqs_json = {str(k): v for k, v in ngram_freqs.items()}
       # affected_dict_json = {str(k): {str(k_a): v_a for k_a, v_a in v.items()} for k, v in affected_dict.items()}
        utils.log_w_wandb(logger=logger,message=f"ngrams freqs & affected dict saved to {utils.cfg_to_filename(cfg)}")
        utils.save_pickle(path_freqs, ngram_freqs)
        utils.save_pickle(path_affected, affected_dict)

    new_vocab = find_new_vocab(ngram_freqs, affected_dict, current_vocab, final_vocab_size)
    return new_vocab

if __name__ == '__main__':
    model_name = 'mistralai/Mistral-7B-Instruct-v0.2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
   # vocab_size = 25000
   # max_n = 4
   # min_freq_affected = 2
   # current_vocab = list(tokenizer.get_vocab().keys())

  #  tokenized_texts = preproccess_texts(data['train']['text'][:10000], tokenizer)
  #  save_json(os.path.join(outputs_folder, 'tokenized_texts.json'), {'tokenized_texts': tokenized_texts})
  #  tokenized_texts = load_json(os.path.join(outputs_folder, 'tokenized_texts.json'))['tokenized_texts']
#
  #  ngram_freqs, affected_dict = get_affected_ngrams(tokenized_texts, max_n, vocab_size, min_freq_affected)
#
  #  ngram_freqs_json = {str(k): v for k, v in ngram_freqs.items()}
  #  affected_dict_json = {str(k): {str(k_a): v_a for k_a, v_a in v.items()} for k, v in affected_dict.items()}
  #  save_json(os.path.join(outputs_folder, 'ngram_freqs.json'), {'ngram_freqs': ngram_freqs_json})
  #  save_json(os.path.join(outputs_folder, 'affected_dict.json'), {'affected_dict': affected_dict_json})
  #  ngram_freqs = load_json(os.path.join(outputs_folder, 'ngram_freqs.json'))['ngram_freqs']
  #  ngram_freqs = {eval(k): v for k, v in ngram_freqs.items()}
  #  affected_dict = load_json(os.path.join(outputs_folder, 'affected_dict.json'))['affected_dict']
  #  affected_dict = {eval(k): {eval(k_a): v_a for k_a, v_a in v.items()} for k, v in affected_dict.items()}
#
  #  new_vocab = find_new_vocab(ngram_freqs, affected_dict, current_vocab, vocab_size)
#
  #  new_vocab = {str(k): v for k, v in new_vocab.items()}
  #  save_json(os.path.join(outputs_folder, 'new_vocab.json'), {'new_vocab': new_vocab})
  #  for token, freq in new_vocab.items():
  #      print(f"{token}: {freq}")
  #  debug = True