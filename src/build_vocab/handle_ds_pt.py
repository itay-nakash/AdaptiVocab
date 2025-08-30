import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.constants import *
import sys
from collections import defaultdict
from datasets import Dataset, load_dataset
import pickle
import re
import json
from typing import Dict
from datetime import datetime
from utils.constants import *
import utils.utils_funcs as utils


# HANDEL DATASETS FOR TOKENIZERS TRAINING

dataset_text_keys = {
    "glue": 'sentence',  
    "cnn_dailymail": ['article','highlights'],
    "medmcqa": ['question','opa','opb','opc','opd'],
    "medal": ['text','label'],
} 


def get_target_ds(cfg:Dict) -> Dataset:
    text_cols=dataset_text_keys.get(cfg['target_corpus_name'], 'text')
    pretrain_courps_name = cfg['target_corpus_name']
    
    # load according to the ds configs:
    if pretrain_courps_name == 'cnn_dailymail':
        pretrain_corpus = load_dataset(pretrain_courps_name, '3.0.0')
    else:
        if 'machelreid/m2d2' in pretrain_courps_name:
            #rest of courpus name:
            config= pretrain_courps_name.split('/')[1][len('m2d2_'):]
            pretrain_courps_name = 'machelreid/m2d2'
            pretrain_corpus = load_dataset(pretrain_courps_name,config)
        else:
            pretrain_corpus = load_dataset(pretrain_courps_name)
        
    return pretrain_corpus

def get_filtered_ds(ds:Dataset) -> List[str]:
    text_cols='text' #TODO: might need to adjust in future
    filtered_ds=[text for text in ds[text_cols] if len(text.split()) >= MIN_EXAMPLE_LEN]
    return filtered_ds

def get_tokenized_ds(cfg:Dict,ds:Dataset,tokenizer) -> Dataset:
    # create / name the folder to save in:
    save_folder = PROCESSED_DATA_PATH
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    # get file name and ds text cols:
    safe_file_name = utils.get_safe_filename(f"{cfg['target_corpus_name']}_{cfg['original_tokenizer']}_tokenized.pkl")
    file_path = os.path.join(save_folder, safe_file_name)
    
    # if exists - load the processed corpus:
    if os.path.isfile(file_path):
        with open(file_path, 'rb') as file:
            pretrain_corpus_tokenized = pickle.load(file)
    else:
        text_cols=dataset_text_keys.get(cfg['target_corpus_name'], 'text')
        pretrain_corpus_tokenized=[]
        for text in ds[text_cols]:
            if len(text.split()) < MIN_EXAMPLE_LEN:
                continue
            tokenized_text = tokenizer.tokenize(text)
            pretrain_corpus_tokenized.append(tokenized_text)
        with open(file_path, 'wb') as file:
            pickle.dump(pretrain_corpus_tokenized, file)
            
    return pretrain_corpus_tokenized  

def get_token_freqs_dict(tokenized_corpus:List[List[str]],big_corpus_tokenized:List[List[str]]=[],alpha:float=EPSILON)-> Dict[str,int]:
    tok_freqs=defaultdict(int)
    for text in tokenized_corpus:
        for tok in text:
            tok_freqs[tok] += 1
    if big_corpus_tokenized is not None: 
        raise NotImplementedError("adding big_corpus to stats is not implemented yet")   
        # assert make sure its not overfloez
        new_word_freqs_sum=sum(word_freqs.values())
        old_word_freqs_sum=sum(old_word_freqs.values())
        assert new_word_freqs_sum != float('inf'), "Sum of new_word_freqs has overflowed!"
        assert old_word_freqs_sum != float('inf'), "Sum of old_word_freqs has overflowed!"
        normalize_amount= new_word_freqs_sum/old_word_freqs_sum
        old_word_freqs_normalized = {k: v*normalize_amount for k, v in old_word_freqs.items()}
        for tok in old_word_freqs:
            if tok not in tok_freqs:
                tok_freqs[tok] = alpha* old_word_freqs_normalized[tok]
            else:
                tok_freqs[tok] = (1-alpha)*tok_freqs.get(tok, 0) + alpha*old_word_freqs_normalized[tok]
    else:
        return tok_freqs
    
def get_tokens_freqs(cfg:Dict,tokenized_corpus:List[List[str]],big_corpus_tokenized:List[List[str]])-> Dict[str,int]:
    path = os.path.join(DATASET_STATS_PATH, f"{utils.cfg_to_tokenfreq_name(cfg)}_tokens_freqs.pkl")
    try:
        with open(path, 'rb') as file:
            tokens_freqs = pickle.load(file)
        utils.log_w_wandb(logger=logger, message=f"Tokens stats loaded from {path}")
    except FileNotFoundError:
        tokens_freqs = get_token_freqs_dict(tokenized_corpus=tokenized_corpus, big_corpus_tokenized=big_corpus_tokenized, alpha=cfg.get('alpha',0))
        utils.log_w_wandb(logger=logger, message=f"Tokens stats saved to {path}")
        # Ensure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as file:
            pickle.dump(tokens_freqs, file)

    return tokens_freqs