import os
import json
import random
import numpy as np
import torch
import sys
from collections import defaultdict
from datasets import Dataset, load_dataset
import os
import pickle
import re
import json
from typing import Dict
from datetime import datetime
from transformers import AutoTokenizer
from utils.constants import *
import hashlib


dir_hash_mapping={}
# general utils:

def set_seed(seed):
    """
    Sets the seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def is_debug_mode():
    """ Check if the script is being debugged. """
    return False
    if __debug__:
        return True
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace is None:
        return False
    return gettrace() is not None 

def get_config():
    wandb.init(project="tokenizers-compare", entity="smooth_language")
    cfg=wandb.config
    run_name =cfg_to_filename(cfg)
    wandb.run.name = run_name
    wandb.run.save()

    return cfg

def save_json(file_path: str, data: Dict):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def save_pickle(file_path: str, data: dict):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:  # Note 'wb' for writing in binary mode
        pickle.dump(data, f)

def load_pickle(file_path: str) -> dict:
    with open(file_path, 'rb') as f:  # Note 'rb' for reading in binary mode
        return pickle.load(f)
        
def load_data_from_directory(directory_path: str):
    """
    Load all data files from a specified directory.

    Parameters:
    directory_path (str): Path to the directory containing the data files.

    Returns:
    dict: A dictionary containing the loaded data, keyed by file names.
    """
    data = {}
    for filename in os.listdir(directory_path):
        if filename.endswith('.pkl'):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'rb') as file:
                data[filename] = pickle.load(file)
    return data

# logs:


def log_w_wandb(logger, message, level='info', enable_wandb=True, **kwargs):
    """
    Logs messages to both standard logging and wandb if enabled.

    :param logger: The logger object (from Python logging module).
    :param message: The message to log.
    :param level: The logging level ('debug', 'info', 'warning', 'error', 'critical').
    :param enable_wandb: Whether to log to wandb (default: True).
    :param kwargs: Additional key-value pairs to log in wandb.
    """
    level_dict = {
        'debug': logger.debug,
        'info': logger.info,
        'warning': logger.warning,
        'error': logger.error,
        'critical': logger.critical
    }

    # Standard logging
    log_func = level_dict.get(level, logger.info)
    log_func(message)

# names utils:
def cfg_to_filename(cfg:Dict)->str:
    cfg_as_filename = ''
    for k, v in cfg.items():
        if 'pretrain_tokenizer' in k:
            cfg_as_filename += f"{k}:{v}_"
        if 'max' in k:
            continue
        if 'pretrain' in k:
            continue
        if 'analyze' in k:
            continue
        if 'tokenizer_path' in k:
            continue
        cfg_as_filename += f"{k}:{v}_"
    cfg_as_filename=get_safe_filename(cfg_as_filename)
    return cfg_as_filename


def filename_to_cfg(filename: str) -> Dict:
    # Remove the safe filename encoding if applied
    # Assuming get_safe_filename function replaces non-alphanumeric characters, reverse it
    # This step depends on what get_safe_filename does exactly. Adjust as necessary.
    filename = filename.replace('_', ':')
    
    cfg = {}
    parts = filename.split(':')
    
    i = 0
    while i < len(parts):
        if i + 1 < len(parts):
            key = parts[i]
            value = parts[i + 1].split('_')[0]  # Split by _ to handle multiple key-value pairs
            cfg[key] = value
            i += 2
        else:
            break
    
    return cfg

        
def cfg_to_tokenfreq_name(cfg:Dict)->str:
    return get_safe_filename(f"tokenizer_{cfg['original_tokenizer']}_alpha{cfg.get('alpha',0)}_targetcourpus_{cfg['target_corpus_name']}_token_freqs")

def cfg_to_ngramfreq_name(cfg:Dict)->str:
    return get_safe_filename(f"tokenizer_{cfg['original_tokenizer']}_alpha{cfg.get('alpha',0)}_targetcourpus_{cfg['target_corpus_name']}_ngram_freqs")

def get_safe_filename(file_name:str):
    return re.sub(r'[^\w\-_\.]', '_', file_name)

def save_hash_mapping(mapping):
    """Helper function to save the dictionary to a JSON file."""
    with open(HASH_FILE_PATH, 'w') as file:
        json.dump(mapping, file)

def load_hash_mapping():
    """Helper function to load the dictionary from a JSON file."""
    if os.path.exists(HASH_FILE_PATH):
        with open(HASH_FILE_PATH, 'r') as file:
            return json.load(file)
    else:
        assert False, f"Error: {HASH_FILE_PATH} does not exist"
        return {}  # Return an empty dictionary if the file does not exist

def hash_output_dir(output_dir: str) -> str:
    # Load the existing mapping from the file
    try:
        dir_hash_mapping = load_hash_mapping()
    except:
        print(f"Error: {HASH_FILE_PATH} does not exist or does not contain a valid JSON object")
        dir_hash_mapping = {}
    
    # Generate a SHA-256 hash of the output directory
    hash_object = hashlib.sha256(output_dir.encode())
    
    # Create a shorter hashed directory name with a prefix
    hashed_dir = f"model_{hash_object.hexdigest()[:10]}"
    dir_hash_mapping[hashed_dir] = output_dir

    # Save the updated mapping back to the JSON file
    save_hash_mapping(dir_hash_mapping)

    return hashed_dir

def hashed_dir_to_cfg(model_hash: str) -> Dict:
    # Load the existing mapping from the file
    dir_hash_mapping = load_hash_mapping()
    
    # Look up the original output directory using the hashed directory name
    if model_hash not in dir_hash_mapping:
        raise ValueError(f"Error: Hash {model_hash} does not exist in the mapping file")
    
    original_output_dir = dir_hash_mapping[model_hash]
    
    # Extract the filename from the original output directory
    filename = os.path.basename(original_output_dir)
    
    # Convert the filename back to the configuration dictionary
    cfg = filename_to_cfg(filename)
    
    return cfg


  
# parse utils:
def get_ds_config_names(ds_name:str)->Tuple[str,str]:
    if 'm2d2' in ds_name:
        new_ds_name=ds_name.split('/')[0]+'/m2d2'
        config_name=ds_name.split('/')[1][len('m2d2_'):]
    return new_ds_name,config_name


# tokenize_utils

def tokenize_exect_text(tokenizer,word):
    word_to_tok=UNIQE_TOKEN_TO_ADD+word
    tok_word=tokenizer.tokenize(word_to_tok)[2:]
    word_as_tokenized=word.replace(' ','▁')
    flatten_text=''
    for x in tok_word:
        if type(x)==tuple:
            for y in x:
                flatten_text+=y
        else:
            flatten_text+=x
    assert word_as_tokenized==flatten_text, f"Error in tokenizing {word} to {tok_word}"
    return tok_word

def encode_exect_word(tokenizer,word):
    word_to_tok=UNIQE_TOKEN_TO_ADD+word
    tok_word=tokenizer.encode(word_to_tok,add_special_tokens=False)[2:]
    tokens=tokenizer.convert_ids_to_tokens(tok_word)
    decoded_word = ''.join(tokens)
    assert decoded_word==word, f"Error in tokenizing {word} to {tok_word}"
    return tok_word

def assert_logits_alight(student_ids:torch.Tensor,logits:List[int],student_tokens_mapping_reverse_dict_debug,original_sentece:str):
    MINIMUM_RATIO=0.05
    same_as_student_ids=0
    same_as_prev=0
    same_as_next=0
    for idx,logit in enumerate(logits):
        # get max value without changing original logit:
        max_val=max(logit, key=logit.get)
        #second max:
       # second_max_val = sorted(logit, key=logit.get)[-2] if len(logit)>=2 else max_val
        
        #compare to student_ids:
        student_token = student_tokens_mapping_reverse_dict_debug[student_ids[idx].item()]
        if max_val==student_token:
            same_as_student_ids+=1
        next_student_token = student_tokens_mapping_reverse_dict_debug[student_ids[idx+1].item()] if idx+1<len(student_ids) else 'None'
        previouse_student_token = student_tokens_mapping_reverse_dict_debug[student_ids[idx-1].item()] if idx-1>=0 else 'None'
        if max_val==next_student_token:
            same_as_next+=1
        if max_val==previouse_student_token:
            same_as_prev+=1
   # assert same_as_student_ids/len(logits)>MINIMUM_RATIO, f"Only {same_as_student_ids/len(logits)} of the logits are the same as the student_ids"
    #assert same_as_student_ids>same_as_next, f"Assert that logits alight with student_ids, has {same_as_student_ids/len(logits)} vs {same_as_next/len(logits)} next"
    #assert same_as_student_ids>same_as_prev, f"Assert that logits alight with student_ids, has {same_as_student_ids/len(logits)} vs {same_as_prev/len(logits)} prev"    
    return (same_as_student_ids>same_as_next) and (same_as_student_ids>same_as_prev) and (same_as_student_ids/len(logits)>MINIMUM_RATIO)
    #log_w_wandb(logger, f"____________________________________________________________")
    #log_w_wandb(logger, f"Assert that logits alight with student_ids, has {same_as_student_ids/len(logits)}")
    #log_w_wandb(logger, f"Assert that logits alight with student_ids, has  {same_as_next/len(logits)} for next token")
    #log_w_wandb(logger, f"Assert that logits alight with student_ids, has  {same_as_prev/len(logits)} for prev token")
# datastes utils:
def correct_nesting_for_dataset(dataset:Dataset)->Dataset:
    """Applies the flattening function to the 'input_ids', 'attention_mask', and 'labels' of the dataset."""
    def flatten_if_nested(list_or_nested_list):
        """Flattens a nested list if necessary, otherwise returns the list as is."""
        if all(isinstance(elem, list) for elem in list_or_nested_list):
            # Assuming one level of nesting that needs to be flattened.
            return [item for sublist in list_or_nested_list for item in sublist]
        return list_or_nested_list
    def correct_nesting(examples):
        for feature in ['input_ids', 'attention_mask', 'labels']:
            if feature in examples:
                examples[feature] = [flatten_if_nested(example) for example in examples[feature]]
        return examples
    return dataset.map(correct_nesting, batched=True)

def convert_to_pt(example:Dict[str,List[int]])->Dict[str,torch.Tensor]:
    # Convert each field in the batch from a list to a tensor
    example['input_ids'] = torch.tensor(example['input_ids'])
    example['attention_mask'] = torch.tensor(example['attention_mask'])
    example['labels'] = torch.tensor(example['labels'])
    return example




# ngrams dicts utils:



def combine_ngram_hist(ngram_hist:Dict[str,int],curr_ngram_hist:Dict[str,int])->Dict[str,int]:
    for ngram in curr_ngram_hist:
        if ngram_hist:
            ngram_hist[ngram] += curr_ngram_hist[ngram]
        else:
            ngram_hist = curr_ngram_hist
    return ngram_hist


def find_latest_checkpoint(model_dir: str)->str:
    """
    Find the checkpoint directory with the highest number in its name, indicating the latest training checkpoint.

    Parameters:
    model_dir (str): Path to the directory containing all model checkpoints.

    Returns:
    str: Path to the latest checkpoint directory.
    """
    checkpoint_paths = [d for d in os.listdir(model_dir) if d.startswith("checkpoint-")]
    # Extract step numbers from the directory names and find the maximum
    latest_checkpoint = max(checkpoint_paths, key=lambda x: int(x.split('-')[1]), default=None)
    return os.path.join(model_dir, latest_checkpoint) if latest_checkpoint else None

def hash_to_tokenizer_dir(hash:str)->str:
    
    return os.path.join(SAVED_PATCH_TOKENIZER_PATH, hash)


def test_model_single_example(text:str,model,tokenizer)-> Tuple[str,Dict]:
    cur_input=tokenizer(text, return_tensors='pt')
    cur_input=cur_input['input_ids']
    cur_input= torch.tensor(cur_input)
    cur_input=cur_input.to(model.device)
    pad_token_id = tokenizer.existing_tokenizer.pad_token_id # works only on patch tokenizer
    output_sequences = model.generate(
    input_ids=cur_input,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    early_stopping=True,
    do_sample=True,
    num_beams=1, 
    temperature=1.0,
    max_length=512,
    pad_token_id=pad_token_id)
    
    generated_sequence,curr_ngram_hist = tokenizer.decode(output_sequences[0], skip_special_tokens=True,count_ngrams=True)
    return generated_sequence,curr_ngram_hist


def model_name_to_tokenizer_path(model_name:str)->str:
    # Plaster - need propar fix
    if 'vanila' in model_name:
        return '/data/home/itay.nakash/patch_tokenizer/saved_patch_tokenizers/_vanila_mistralai/Mistral-7B-v0.1_Natural_and_physical_sciences__Earth_sciences'
    
    # /data/home/itay.nakash/patch_tokenizer/saved_patch_tokenizers/original_tokenizer_mistralai_Mistral-7B-v0.1_target_corpus_name_machelreid_m2d2_Natural_and_physical_sciences__Earth_sciences_num_to_add_10000_num_to_remove_10000_ngram_k_4_
    if 'mistralai_Mistral-7B-Instruct-v0.2' in model_name:
        original_tokenizer_str='original_tokenizer_mistralai_Mistral-7B-Instruct-v0.2'
    elif 'mistralai_Mistral-7B-v0.1' in model_name:
        original_tokenizer_str='original_tokenizer_mistralai_Mistral-7B-v0.1'
    else:
        raise ValueError(f"Existing tokenizer couldn't be found in {model_name}")
    if 'Natural_and_physical_sciences__Earth_sciences' in model_name:
        target_corpus_name = '_target_corpus_name_machelreid_m2d2_Natural_and_physical_sciences__Earth_sciences'
    else:
        raise ValueError(f"Target corpus couldn't be found in {model_name}")
    if 'num_to_add_10000' in model_name:
        num_to_add = '_num_to_add_10000'
    if 'num_to_remove_10000' in model_name:
        num_to_remove = '_num_to_remove_10000'
    if 'ngram_k_4' in model_name:
        ngram_k = '_ngram_k_4_'
    if 'date_major_up_0306' in model_name:
        date_major_up = 'date_major_up_0306_'
    else:
        date_major_up = ''
    tokenizer_dir_path = os.path.join(SAVED_PATCH_TOKENIZER_PATH, f'{original_tokenizer_str}{target_corpus_name}{num_to_add}{num_to_remove}{ngram_k}{date_major_up}')
    print(f"tokenizer_dir_path: {tokenizer_dir_path}")
    return tokenizer_dir_path


def path_name_to_cfg(path_name:str)->Dict:
    # Initialize an empty dictionary to store the configuration
    cfg = {}
    
    # Remove any unsafe filename remnants or extensions (e.g., ".txt", ".json")
    # Assuming the path_name might include such extensions and prepended directories
    # We just want the last part after the last slash which would be the filename
    filename = path_name.split('/')[-1].split('.')[0]

    # Split the filename by underscores to separate different key-value pairs
    key_value_pairs = filename.split('_')
    
    # Iterate through each key-value pair
    for pair in key_value_pairs:
        # Split the pair by the colon to separate the key and the value
        if ':' in pair:
            key, value = pair.split(':', 1)  # Only split at the first colon
            cfg[key] = value
    
    return cfg

def get_eval_ds_path(tokenizer_path:str)->Dict:
    eval_datasets={'validation':{},'test':{}}
    
    eval_datasets['validation'] = {
    "domain_qa_context": f"{tokenizer_path}/tokenized_tasks/validation/domain_qa_context_tokenized.json",
    "domain_qa_no_context": f"{tokenizer_path}/tokenized_tasks/validation/domain_qa_no_context_tokenized.json",
    "domain_qa_no_context_long": f"{tokenizer_path}/tokenized_tasks/validation/domain_qa_no_context_long_tokenized.json",
    "domain_qa_context_over_train": f"{tokenizer_path}/tokenized_tasks/validation/domain_qa_context_over_train_tokenized.json",
    "domain_qa_no_context_over_train": f"{tokenizer_path}/tokenized_tasks/validation/domain_qa_no_context_over_train_tokenized.json",
    "perplexity_data": f"{tokenizer_path}/tokenized_tasks/validation/perplexity_data.json",
    "generated_data": f"{tokenizer_path}/tokenized_tasks/validation/generated_data.json",
    "arc_qa": f"{tokenizer_path}/tokenized_tasks/validation/qa/arc_qa.json",
    "ecqa_qa": f"{tokenizer_path}/tokenized_tasks/validation/qa/ecqa_qa.json",
    "openbookqa_qa": f"{tokenizer_path}/tokenized_tasks/validation/qa/openbookqa_qa.json",
    "qasc_qa": f"{tokenizer_path}/tokenized_tasks/validation/qa/qasc_qa.json"
}
    
    eval_datasets['test'] = {
    "domain_qa_context": f"{tokenizer_path}/tokenized_tasks/test/domain_qa_context_tokenized.json",
    "domain_qa_no_context": f"{tokenizer_path}/tokenized_tasks/test/domain_qa_no_context_tokenized.json",
    "domain_qa_no_context_long": f"{tokenizer_path}/tokenized_tasks/test/domain_qa_no_context_long_tokenized.json",
    "domain_qa_context_over_train": f"{tokenizer_path}/tokenized_tasks/test/domain_qa_context_over_train_tokenized.json",
    "domain_qa_no_context_over_train": f"{tokenizer_path}/tokenized_tasks/test/domain_qa_no_context_over_train_tokenized.json",
 #   "perplexity_data": f"{tokenizer_path}/tokenized_tasks/test/perplexity_data.json",
    "generated_data": f"{tokenizer_path}/tokenized_tasks/test/generated_data.json",
    "arc_qa": f"{tokenizer_path}/tokenized_tasks/test/qa/arc_qa.json",
    "ecqa_qa": f"{tokenizer_path}/tokenized_tasks/test/qa/ecqa_qa.json",
    "openbookqa_qa": f"{tokenizer_path}/tokenized_tasks/test/qa/openbookqa_qa.json",
    "qasc_qa": f"{tokenizer_path}/tokenized_tasks/test/qa/qasc_qa.json"
    }
    return eval_datasets


# kd debbug:
# seq_len X vocab_size
def check_logits_align(teacher_logits:torch.Tensor,student_logits:torch.Tensor,c_step:int):
    NUMBER_OF_TOPK=5
    top_preds_t = torch.topk(teacher_logits[0], NUMBER_OF_TOPK, dim=-1).indices
    top_preds_s = torch.topk(student_logits[0], NUMBER_OF_TOPK, dim=-1).indices
    # check how much time it aggress:
    same_as_teacher=0
    same_as_prev=0
    same_as_next=0
    for idx,s_preds in enumerate(top_preds_s):
        # if someone from there indexes group is alike add
        for s_pred in s_preds:
            if s_pred in top_preds_t[idx]:
                same_as_teacher+=1
            if idx-1>0 and s_pred in top_preds_s[idx-1]:
                same_as_prev+=1
            if idx+1<len(top_preds_s) and s_pred in top_preds_s[idx+1]:
                same_as_next+=1
    #print(f"same_as_teacher: {round(same_as_teacher/(len(top_preds_t)*NUMBER_OF_TOPK),3)}")
    #print(f"same_as_prev: {round(same_as_prev/(len(top_preds_t)*NUMBER_OF_TOPK),3)}")
    #print(f"same_as_next: {round(same_as_next/(len(top_preds_t)*NUMBER_OF_TOPK),3)}")
    same_as_teacher = round(same_as_teacher/(len(top_preds_t)*NUMBER_OF_TOPK),3)
    same_as_prev = round(same_as_prev/(len(top_preds_t)*NUMBER_OF_TOPK),3)
    same_as_next = round(same_as_next/(len(top_preds_t)*NUMBER_OF_TOPK),3)
    if wandb.run is not None:
        wandb.log(data={"c_step":c_step,"same_as_teacher":same_as_teacher,"same_as_prev":same_as_prev,"same_as_next":same_as_next})
    return same_as_teacher,same_as_prev,same_as_next
    #assert same_as_teacher>same_as_prev, f"Assert that logits alight with teacher, has {same_as_teacher} vs {same_as_prev} prev"
    #assert same_as_teacher>same_as_next, f"Assert that logits alight with teacher, has {same_as_teacher} vs {same_as_next} next"


