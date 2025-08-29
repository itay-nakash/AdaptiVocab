import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.constants import *
import re
import torch
from collections import defaultdict
from transformers import AutoTokenizer
import torch
import pickle
import utils.utils_funcs as utils

class PatchTokenizer():
    def __init__(self, existing_tokenizer_name: str, removed_tokens:Dict[str,List[str]]={}, ngram_dict:Dict[Tuple,int]={}):
        self.existing_tokenizer = AutoTokenizer.from_pretrained(existing_tokenizer_name)
        self.bos_token = self.existing_tokenizer.bos_token 
        self.eos_token = self.existing_tokenizer.eos_token 
        self.bos_token_id = self.existing_tokenizer.bos_token_id
        self.eos_token_id = self.existing_tokenizer.eos_token_id
        self.pad_token = self.eos_token if self.existing_tokenizer.pad_token_id == None else self.existing_tokenizer.pad_token
        self.pad_token_id = self.existing_tokenizer.eos_token_id  if self.existing_tokenizer.pad_token_id == None else self.existing_tokenizer.pad_token_id
        self.removed_tokens = removed_tokens
        self.ngram_dict = ngram_dict
        self.id_converter = None

    # ------ Basic tokenization functions:  ------ 
    def tokenize(self, text:str):
    # Use the existing tokenizer to tokenize the text
        if type(text)==list:
            assert len(text)==1, "Patch_Tokenizer can only tokenize one text at a time"
            text = text[0] 
        tokens = self.existing_tokenizer.tokenize(text)
        upd_tokens=[]

        # remove reduced tokens:
        for token in tokens:
            if token in self.removed_tokens.keys():
                upd_tokens+=self.removed_tokens[token] # extend the tokens list
            else:
                upd_tokens.append(token)
        tokens = upd_tokens

        # Replace tokens with ngrams where applicable, in the order of scores
        upd_tokens=[]
        i = 0
        while i < len(tokens):
            replaced = False
            for (ngram, score) in self.ngram_dict.items():
                # No need to split since ngram is already a tuple of tokens
                if tuple(tokens[i:i+len(ngram)]) == ngram:
                    # Replace with ngram token_id and move index past this ngram
                    upd_tokens.append(ngram)
                    i += len(ngram) - 1 
                    replaced = True
                    break 
            if not replaced:
                # No ngram match found, keep original token
                upd_tokens.append(tokens[i])
            i += 1
            
        return upd_tokens
    
    
    def __call__(self, texts:List[str],  padding:bool=False,truncation=False,
                 max_length:int=sys.maxsize,return_tensors:str='no')->Dict:
        if type(texts)==list:
            assert len(texts)==1, "Patch_Tokenizer can only tokenize one text at a time"
            texts = texts[0]
        # Tokenize the text using the tokenizer function to get tokens as strings
        tokens = self.tokenize(texts)

        converted_token_ids = []

        # Iterate over each token to convert it according to the id_converter
        for token in tokens:
            if token==self.existing_tokenizer.unk_token:
                converted_token_ids.append(self.existing_tokenizer.unk_token_id)
                continue
            assert token in self.id_converter, f"{token} token is not in the convertor dict"
            # Retrieve the conversion info
            conversion_info = self.id_converter[token]
            old_indices = conversion_info['old_indices']
            new_indices = conversion_info['new_index']

            converted_token_ids.append(new_indices)
        if truncation:# token trimming:
            converted_token_ids = converted_token_ids[:max_length]
        attention_mask = ([1] * len(converted_token_ids))
        if padding:
            raise NotImplementedError("padding is not implemented yet")
            converted_token_ids.extend([self.existing_tokenizer.pad_token_id] * (max_length - len(converted_token_ids)))
            attention_mask.extend([0] * (max_length - len(converted_token_ids)))
        
        #if self.decode(converted_token_ids) != texts:
        #    if "</s>"!=texts:
        #        texts = re.sub(r"<s>", "", texts)
        #        texts = re.sub(r"</s>", " ", texts)
        #        assert self.decode(converted_token_ids) == texts, f"Decoded text does not match original text: {self.decode(converted_token_ids)} != {texts}"
        if return_tensors=='pt':
            return {"input_ids": torch.tensor(converted_token_ids).unsqueeze(0).to('cuda'), "attention_mask": torch.tensor(attention_mask).unsqueeze(0).to('cuda')}
        else:
            return {"input_ids": converted_token_ids, "attention_mask": attention_mask}

    def decode(self, token_ids:torch.Tensor,skip_special_tokens=True,count_ngrams=False):
        # Convert token IDs to list
        token_ids_list = token_ids.tolist() if isinstance(token_ids, torch.Tensor) else token_ids
        tokens_list= [self.id_to_token_dict[token_id] for token_id in token_ids_list]
        original_tokens_ids = []
        for token in tokens_list:
            if token in self.id_converter:
                old_indices = self.id_converter[token]['old_indices']
                if isinstance(old_indices, list):
                    original_tokens_ids.extend(old_indices)  # Extend the list if old_indices is a list
                else:
                    original_tokens_ids.append(old_indices)  # Append the single int if old_indices is not a list
            else:
                original_tokens_ids.append(50265)  # Append the default value if token is not in id_converter

        decoded_str = self.existing_tokenizer.decode(original_tokens_ids,skip_special_tokens=skip_special_tokens)
        if count_ngrams:
            ngram_hist = defaultdict(int)
            for token in tokens_list:
                if type(token)==tuple:
                    ngram_hist[token] += 1            
            return decoded_str, ngram_hist
        else:
            return decoded_str

    def convert_ids_to_tokens(self,input_ids) -> List:
        return [self.id_to_token_dict[token_id.item()] for token_id in input_ids]

    def get_vocab(self)->Dict[str,int]:
        return self.token_to_id_dict 
   
    def save_pretrained(output_dir:str, *args, **kwargs):
        utils.log_w_wandb(logger=logger,message=f" ----------------- Tokenizer 'saved_pretrained' function was called but not implemented in PatchTokenizer -----------------",logging_level='info')
        utils.log_w_wandb(logger=logger,message=f" --------------------------------------------------- Skipping --------------------------------------------------------------------",logging_level='info')
    # ------  id convertor functions: ------ 
    
    
    def add_converter(self, converter_dict:Dict[str,Dict[str, Union[List[int], int]]]):
        """Adds a converter dictionary to the tokenizer.
        
        Args:
            converter_dict (dict): A dictionary mapping old IDs to new IDs.
        """
        self.id_converter = converter_dict
        self.create_token_to_id_dict()
        self.create_id_to_token_dict()

    def create_token_to_id_dict(self):
        # Create a dictionary mapping each token to its ID
        token_to_id_dict = {}
        for token in self.id_converter:
            assert type(self.id_converter[token]['new_index']) == int, f"Token {token} has multiple new indices"
            token_to_id_dict[token] = self.id_converter[token]['new_index']
        self.token_to_id_dict = token_to_id_dict
        
    def create_id_to_token_dict(self):
        # Create a dictionary mapping each ID to its token
        id_to_token_dict = {}
        for token in self.id_converter:
            assert type(self.id_converter[token]['new_index'])==int, f"Token {token} has multiple new indices"
            assert self.id_converter[token]['new_index'] not in id_to_token_dict, f"Token {token} has multiple new indices"
            id_to_token_dict[self.id_converter[token]['new_index']] = token
            
        self.id_to_token_dict = id_to_token_dict
        
    
    # ------  save & load functions: ------ 
     
     
    def save_model(self, path:str):
        with open(path, 'wb') as file:
            pickle.dump({
                "existing_tokenizer": self.existing_tokenizer, 
                "ngram_dict": self.ngram_dict,
                "removed_tokens": self.removed_tokens,
                "id_converter": self.id_converter,                    
                "token_to_id_dict": self.token_to_id_dict,
                "id_to_token_dict": self.id_to_token_dict
            }, file)
        logger.info(f"Model and ngram dictionary saved to {path}")

    def load_model(self, path:str):
        # Load the tokenizer model and ngram dictionary from the given path
        with open(path, 'rb') as file:
            data = pickle.load(file)
            self.existing_tokenizer = data["existing_tokenizer"]
            self.ngram_dict = data["ngram_dict"],
            self.removed_tokens = data["removed_tokens"]
            self.id_converter = data["id_converter"]
            self.token_to_id_dict = data["token_to_id_dict"]
            self.id_to_token_dict = data["id_to_token_dict"]
        self.logger.info(f"Model and ngram dictionary loaded from {path}")
        
    @staticmethod
    def load_model_from_scratch(path:str ,existing_tokenizer_name: str):
        # Load the model and ngram dictionary from the given path
        with open(path, 'rb') as file:
            data = pickle.load(file)

        # Create a new instance of PatchTokenizer with loaded data
        loaded_tokenizer = PatchTokenizer(
            existing_tokenizer_name=existing_tokenizer_name,  # Placeholder or retrieve from saved data
            ngram_dict=data["ngram_dict"],
            removed_tokens=data["removed_tokens"]
        )
        loaded_tokenizer.id_converter = data["id_converter"]
        loaded_tokenizer.token_to_id_dict = data["token_to_id_dict"]
        loaded_tokenizer.id_to_token_dict = data["id_to_token_dict"]
        logger.info(f"Model and ngram dictionary loaded from {path}")            
        
        
        # edit bos and eos tokens according to values:
        loaded_tokenizer.bos_token_id = loaded_tokenizer(loaded_tokenizer.bos_token)['input_ids'][0]
        loaded_tokenizer.eos_token_id = loaded_tokenizer(loaded_tokenizer.eos_token)['input_ids'][0]
        loaded_tokenizer.pad_token_id = loaded_tokenizer(loaded_tokenizer.pad_token)['input_ids'][0]
        
        # switch bos and eos tokens to the wanted ones:
        return loaded_tokenizer
