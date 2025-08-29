import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.constants import *
import utils.utils_funcs as utils
from transformers import AutoTokenizer
import logging
import json
from copy import deepcopy




logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



def get_removed_tokens_decomposed(tokens_to_remove: List[str],original_tokenizer:AutoTokenizer) -> Tuple[Dict[str, int], Dict[str, int]]:
    
    removed_tokens_merges = get_reduced_tokens_merges(original_tokenizer,tokens_to_remove)
    
    # dont remove tokens that dont have merge rule:
    tokens_to_remove = [token for token in tokens_to_remove if token in removed_tokens_merges]
    
    decomposed_dict = get_new_tokenization(tokens_to_remove,original_tokenizer,removed_tokens_merges)
    
    test_asserts_reduced_tokens(decomposed_dict,original_tokenizer)

    return decomposed_dict



def reduce_vocab(original_tokenizer:AutoTokenizer,tokens_freqs: Dict[str,int],cfg:Dict)->Dict[str,List[str]]:
    
    logger.info(f"Initial model vocab size:{len(original_tokenizer.get_vocab())}, tokens to remove: {cfg['num_to_remove']}")
    

    tokens_to_remove = get_tokens_to_remove(original_tokenizer.get_vocab(), tokens_freqs, cfg['num_to_remove'])
    
    removed_tokens_merges = get_reduced_tokens_merges(original_tokenizer,tokens_to_remove)
    
    # dont remove tokens that dont have merge rule:
    tokens_to_remove = [token for token in tokens_to_remove if token in removed_tokens_merges]
    
    removed_tokens = get_new_tokenization(tokens_to_remove,original_tokenizer,removed_tokens_merges)
    
    test_asserts_reduced_tokens(removed_tokens,original_tokenizer)
    
    return removed_tokens


def get_new_tokenization(tokens_to_remove:List[str],tokenizer:AutoTokenizer,reduced_tok_merges:Dict[str,Tuple])->Dict[str,List[str]]:
    reduced_tok=defaultdict(str)
    for tok in tokens_to_remove:
        assert tok in tokenizer.get_vocab(), f"Token {tok} not in tokenizer vocab"
        assert tok in reduced_tok_merges, f"Token {tok} not in reduced_tok_merges"
        reduced_tok[tok]=rec_find_tokenization(tok,tokenizer,tokens_to_remove,reduced_tok_merges)
    reduced_tok_flat = flatten_reduced_tok(deepcopy(reduced_tok))
    return reduced_tok_flat

def rec_find_tokenization(word:str,tokenizer:AutoTokenizer,tokens_to_remove:List[str],reduced_tok_merges:Dict[str,Tuple[str,str]])->List[str]:
    if word not in tokens_to_remove and word in tokenizer.get_vocab():
        return word
    else:
        original_tokenization = reduced_tok_merges[word]
        assert ''.join(original_tokenization) == word, f"Tokenization mismatch: original token '{word}' does not match tokenized and joined version {original_tokenization}."
        ret_tokenization=[]
        for sub_tk in original_tokenization:
            ret_tokenization.append(rec_find_tokenization(sub_tk,tokenizer,tokens_to_remove,reduced_tok_merges))            
    return ret_tokenization


def flatten_reduced_tok(reduced_tok:Dict)->Dict[str,List[str]]:
    """Flatten all values in the dictionary."""
    def flatten_list(nested_list):
        """Flatten a nested list into a list of lists with no nested lists."""
        result = []
        for item in nested_list:
            if isinstance(item, list):
                # If the item is a list, extend the result by recursively flattening the item
                result.extend(flatten_list(item))
            else:
                # If the item is not a list, simply append it as a new list
                result.append([item])
        return result
    flattened_dict = {}
    for key, value in reduced_tok.items():
        flattened_dict[key] = flatten_list(value)
        flattened_dict[key] = [item for sublist in flattened_dict[key] for item in sublist]
        assert ''.join(flattened_dict[key]) == key, f"Tokenization mismatch: original token '{key}' does not match tokenized and joined version {flattened_dict[key]}."
    return flattened_dict




def get_reduced_tokens_merges(pretrain_tokenizer:AutoTokenizer,tokens_to_remove:List[str])->Dict[str,List[str]]:
    reduced_tokens_rule={}
    model_state = json.loads(pretrain_tokenizer.backend_tokenizer.model.__getstate__())
    merges_list = [tuple(x) for x in model_state['merges']]
    for item in merges_list:
        if len(item)<2:
            merges_list.remove(item)

    for wl, wr in merges_list:
        merge = wl + wr
        if merge in tokens_to_remove and merge not in reduced_tokens_rule:
            reduced_tokens_rule[merge]=[wl,wr]            
    return reduced_tokens_rule
    
    

def test_asserts_reduced_tokens(reduced_tok:Dict[str,List[str]],original_tokenizer:AutoTokenizer):
    for token in reduced_tok.keys():
        crnt_tok=reduced_tok[token]
        assert token in original_tokenizer.get_vocab(), f"Token {token} wasn't in tokenizer vocab"
        assert len(crnt_tok) != 0, f"Token {token} was removed from tokenizer but cant be tokenized (len=0)"
        assert len(crnt_tok) != 1, f"Token {token} was removed from tokenizer but still tokenized by one token"
        assert ''.join(crnt_tok) == token, f"Tokenization mismatch: original token '{token}' does not match tokenized and joined version {crnt_tok}."

  
def get_tokens_to_remove(vocab:Dict[str,int], target_corpus_tokens_freqs:Dict[str,int], n_tokens_to_remove:int)->List[str]:
    vocab_tokens = set(vocab.keys())
    corpus_tokens = set(target_corpus_tokens_freqs.keys())

    tokens_not_in_corpus = vocab_tokens - corpus_tokens
    toks_not_in_corpus_to_rm = [token for token in tokens_not_in_corpus if len(token)>1 and token not in SAVED_TOKENS] #dont remove chars or saved tokens
    
    if len(toks_not_in_corpus_to_rm) >= n_tokens_to_remove:
        utils.log_w_wandb(logger=logger,level='warning',message=f"Found {len(toks_not_in_corpus_to_rm)} tokens not in corpus, but asked to remove just {n_tokens_to_remove} tokens.")
        toks_not_in_corpus_to_rm.sort(key=len, reverse=True) # added this to remove the longest tokens...
        return toks_not_in_corpus_to_rm[:n_tokens_to_remove]
    
    n_remaining_tokens_to_remove = n_tokens_to_remove - len(toks_not_in_corpus_to_rm)
    filtered_tokens_by_length = {token: freq for token, freq in target_corpus_tokens_freqs.items() if len(token) > 1 and token not in SAVED_TOKENS}
    sorted_tokens_by_freq = sorted(filtered_tokens_by_length.items(), key=lambda x: x[1])
    least_frequent_tokens = [token for token, freq in sorted_tokens_by_freq[:n_remaining_tokens_to_remove]]
    
    tok_to_remove = list(toks_not_in_corpus_to_rm) + least_frequent_tokens
    assert len(tok_to_remove) == n_tokens_to_remove, f"Expected to remove {n_tokens_to_remove} tokens, but removed {len(tok_to_remove)} tokens."
    
    return tok_to_remove
 
def update_vocab(target_corpus_tokens_scores,target_corpus_tokens_probs,target_corpus_tokens_freqs,n_tokens_to_remove):
    worst_tokens = target_corpus_tokens_scores[-n_tokens_to_remove:]
    for token in worst_tokens:
        del target_corpus_tokens_scores[token]
        del target_corpus_tokens_probs[token]
        del target_corpus_tokens_freqs[token]
    return target_corpus_tokens_scores,target_corpus_tokens_probs,target_corpus_tokens_freqs

