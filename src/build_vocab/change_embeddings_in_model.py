import os
import sys
from utils.constants import *
import pickle
from transformers import OPTForCausalLM, AutoModel,GPT2Tokenizer
import torch
import utils.utils_funcs as utils
import os

# --------- generate new embeddings: --------- 
def exp_avrg(token_embeddings,big_in_end:bool=False):
    """
    Generates embeddings based on decreasing exponential weights.
    """
    num_tokens = token_embeddings.shape[0]
    if big_in_end:
        weights = [0.5**i for i in range(num_tokens)][::-1]
    else:
        weights = [0.5**i for i in range(num_tokens)]
    weights = torch.tensor([w / sum(weights) for w in weights], device=token_embeddings.device, dtype=token_embeddings.dtype)
    return (token_embeddings * weights.unsqueeze(1)).sum(dim=0)
    

# ---------  create mapping: --------- 
def create_new_inputs_ids_mapping_old(token_to_id_old:Dict[str,int],reduced_tokens:List[str],tokens_to_add:List[str]) -> Tuple[Dict[str,int],Dict[int,str]]:
    assert len(reduced_tokens) == len(set(reduced_tokens)), "reduced_tokens contains duplicates"
    assert len(tokens_to_add) == len(set(tokens_to_add)), "tokens_to_add contains duplicates"
    assert len(reduced_tokens) >= len(tokens_to_add), "reduced_tokens should be bigger than tokens_to_add"
    
    token_to_id_new = {}
    id_to_token_new = {}
    
    for i,(token,old_id) in enumerate(token_to_id_old.items()):
        if token in reduced_tokens:
            continue # tokens that were removed are not in the new tokenizer, they dont have id
        token_to_id_new[token] = i
        id_to_token_new[i] = token

    last_original_token_index = len(token_to_id_new)
    
    for i,token in enumerate(tokens_to_add):
        new_token_idx = i + last_original_token_index
        token_to_id_new[token] = new_token_idx
        id_to_token_new[new_token_idx] = token

    return token_to_id_new,id_to_token_new

def create_new_inputs_ids_mapping(token_to_id_old:Dict[str,int],reduced_tokens:List[str],tokens_to_add:List[str]) -> Tuple[Dict[str,int],Dict[int,str]]:
    assert len(reduced_tokens) == len(set(reduced_tokens)), "reduced_tokens contains duplicates"
    assert len(tokens_to_add) == len(set(tokens_to_add)), "tokens_to_add contains duplicates"
    assert len(reduced_tokens) >= len(tokens_to_add), f"reduced_tokens should be bigger than tokens_to_add, reduced_tokens: {len(reduced_tokens)}, tokens_to_add: {len(tokens_to_add)}"
    
    token_to_id_new = {}
    id_to_token_new = {}
    new_index=0
    for (token,old_id) in token_to_id_old.items():
        if token in reduced_tokens:
            continue # tokens that were removed are not in the new tokenizer, they dont have id
        token_to_id_new[token] = new_index
        id_to_token_new[new_index] = token
        new_index+=1

    for token in tokens_to_add:
        token_to_id_new[token] = new_index
        id_to_token_new[new_index] = token
        new_index+=1

    assert new_index == len(id_to_token_new)
    return token_to_id_new,id_to_token_new

def convert_ngram_to_str(ngram:Tuple[str,...]) -> str:
    new_ngram = []
    for i,tok in enumerate(ngram):
        new_ngram.append(tok.replace('Ġ',' '))
    return ''.join(new_ngram)

def create_inputs_ids_convert_dict(token_to_id_new:Dict[str, int],token_to_id_old:Dict[str, int],added_ngrams:Dict[Tuple[str,...],float],removed_tokens:Dict[str,List[str]],original_tokenizer:GPT2Tokenizer,base_path:str) -> Dict[str,Dict[str, Union[List[int], int]]]:
    # create dict taht convert between the new input ids to the old input ids:
    inputs_ids_convert = {}
    for token,token_id in token_to_id_old.items():
        if token not in token_to_id_new:
            assert token in removed_tokens, f"Token {token} is not in the new tokenizer, but not in removed tokens"
            continue # tokens that were removed are not in the new tokenizer, so we don't need to convert them
        inputs_ids_convert[token]={'old_indices':token_id,'new_index':token_to_id_new[token]} 
    for ngram in added_ngrams:
        #add ngram to the inputs_ids_convert
        ngram_as_str=''.join(ngram)
        input_ids_in_old_vocab = utils.encode_exect_word(original_tokenizer,ngram_as_str) 
        inputs_ids_convert[ngram]={'old_indices':input_ids_in_old_vocab,'new_index':token_to_id_new[ngram]}
    assert len(inputs_ids_convert) == len(token_to_id_old), f"inputs_ids_convert has {len(inputs_ids_convert)} tokens, while the old tokenizer has {len(token_to_id_old)} tokens"

    with open(os.path.join(base_path, 'inputs_ids_convert.pkl'), 'wb') as f:
        pickle.dump(inputs_ids_convert, f)
    
    return inputs_ids_convert    

def create_new_embbeding_matrix(old_embedding_matrix:torch.Tensor,inputs_ids_convert:Dict[str,Dict[str, Union[List[int], int]]],emb_method_str:str=MEAN_EMBEDDING) -> torch.Tensor:
    """
    Create a new embedding matrix from the old embedding matrix and the inputs_ids_convert dictionary.
    The new embedding matrix will be with the same size, filled with zeros in empty places
    """
    hidden_dim = old_embedding_matrix.shape[1]
    vocab_size = len(inputs_ids_convert)
    new_embedding_matrix = torch.zeros((vocab_size, hidden_dim)) #TODO: check how to make smaller size than original to work
    for new_token, indices in inputs_ids_convert.items():
        if type(indices['old_indices'])==int:
            new_embedding_matrix[indices['new_index']] = old_embedding_matrix[indices['old_indices']]
        else:
            assert len(indices['old_indices']) > 0 # old_indices is a list of the ngram tokens indices
            if emb_method_str == MEAN_EMBEDDING:
                new_embedding_matrix[indices['new_index']] = old_embedding_matrix[indices['old_indices']].mean(dim=0).detach()
            if emb_method_str == DECREASING_EXP:
                new_embedding_matrix[indices['new_index']] = exp_avrg(old_embedding_matrix[indices['old_indices']],big_in_end=False).detach()
            if emb_method_str == INCREASING_EXP:
                new_embedding_matrix[indices['new_index']] = exp_avrg(old_embedding_matrix[indices['old_indices']],big_in_end=True).detach()
    return new_embedding_matrix

def save_new_embeddings(inputs_ids_convert:Dict[str,Dict[str, Union[List[int], int]]], embedding_input:torch.Tensor, embedding_output:torch.Tensor, model,base_path:str,suffix:str):
    check_new_embeddings(inputs_ids_convert, embedding_input, model.get_input_embeddings().weight)
    with open(os.path.join(base_path, f'embedding_input_{suffix}.pkl'), 'wb') as f:
        pickle.dump(embedding_input, f)
    with open(os.path.join(base_path, f'embedding_output_{suffix}.pkl'), 'wb') as f:
        pickle.dump(embedding_output, f)


# ---------  asserts / tests: --------- 
def check_new_embeddings(inputs_ids_convert:Dict[str,Dict[str, Union[List[int], int]]], embedding_new:torch.Tensor, embedding_old:torch.Tensor):
    for new_token, indices in inputs_ids_convert.items():
        if type(indices['old_indices'])==int:
            assert torch.all(embedding_new[indices['new_index']] == embedding_old[indices['old_indices']]), f"Embedding for {new_token} is not consistent with the old embeddings"
    print("New embeddings are consistent with the old embeddings")




# ---------  main function: --------- 
def get_modified_emb(removed_tokens:Dict[str,List[str]],added_ngrams:Dict[Tuple,float],model,tokenizer,base_path:str):
    token_to_id_old=tokenizer.existing_tokenizer.get_vocab()
    tokens_to_add = [k for k in added_ngrams.keys()]
    
    token_to_id_new,id_to_token_new = create_new_inputs_ids_mapping(token_to_id_old=token_to_id_old,reduced_tokens=removed_tokens,tokens_to_add=tokens_to_add)
    
    inputs_ids_convert = create_inputs_ids_convert_dict(token_to_id_new=token_to_id_new,token_to_id_old=token_to_id_old,added_ngrams=added_ngrams,
                                                        removed_tokens=removed_tokens,original_tokenizer=tokenizer.existing_tokenizer,base_path=base_path)
    
    embedding_input = model.get_input_embeddings().weight
    embedding_output = model.lm_head.weight
    

    new_embedding_input_exp = create_new_embbeding_matrix(embedding_input,inputs_ids_convert,emb_method_str=INCREASING_EXP)
    new_embedding_output_exp = create_new_embbeding_matrix(embedding_output,inputs_ids_convert,emb_method_str=DECREASING_EXP)
    new_embedding_input_mean = create_new_embbeding_matrix(embedding_input,inputs_ids_convert,emb_method_str=MEAN_EMBEDDING)
    new_embedding_output_mean = create_new_embbeding_matrix(embedding_output,inputs_ids_convert,emb_method_str=MEAN_EMBEDDING)
    
    save_new_embeddings(inputs_ids_convert, new_embedding_input_exp, new_embedding_output_exp,model, base_path,EXP_EMB)
    
    save_new_embeddings(inputs_ids_convert, new_embedding_input_mean, new_embedding_output_mean,model, base_path,MEAN_EMB)
    
    return inputs_ids_convert
    
    