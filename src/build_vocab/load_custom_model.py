import gc
import csv
import os
import sys
import pandas as pd
import evaluate.QA.eval
import pickle
import torch
import pickle
import utils.utils_funcs as utils
import build_vocab.change_embeddings_in_model as change_embeddings_in_model
from typing import Dict, List, Union,Tuple
from transformers import (
    AutoTokenizer,AutoModelForCausalLM
)
from peft import get_peft_model, LoraConfig, TaskType
import re
from utils.constants import *
import ft.dp_train
from build_vocab.patch_tokenizer import PatchTokenizer


# THIS CODE LOAD CUSTOM MODELS, EDIT MODELS TO THE MODIFIED VERSION, AND GENERATE WITH LOADED MODEL

add_sweep_name_dict={'model_9a0947fda9':'model_9a0947fda9_sweep_golden-sweep-1',
                'model_228dc46e11':'model_228dc46e11_sweep_breezy-sweep-4',
                'model_7932fa1f09':'model_7932fa1f09_sweep_colorful-sweep-3',
                'model_a9d3237cc1':'model_a9d3237cc1_sweep_rosy-sweep-5',
                'model_bb62fa2388':'model_bb62fa2388_sweep_hearty-sweep-4_used_for_models1',
                'model_534cab487f':'model_534cab487f_sweep_wobbly-blaze-469',
                'model_d562da1be8':'model_d562da1be8_sweep_helpful-brook-493(vanila_pysics)',
                'model_5f0bfb930f':'model_5f0bfb930f_sweep_misty-planet-503',
                'model_4215bb3fa7':'model_4215bb3fa7_sweep_peachy-deluge-498',
                'model_5f0bfb930f':'model_5f0bfb930f_sweep_misty-planet-503',
                'model_a9ac325c9e':'model_a9ac325c9e_sweep_cosmic-oath-546',}

def load_custom_model(model_name:str,emb_method:str,patch_tokenizer_path:str=None,
                      load_checkpoint:bool=False,use_lora=False,unfroz:Tuple[int,int]=(-1,-1)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def modify_model(model , data: Dict[str, object],emb_method:str):
        """
        Modify the model using the loaded data.

        Parameters:
        model: The model to be modified.
        data: The data used for modifying the model, containing embeddings and token mappings.

        Returns:
        The modified model.
        """
        device_input = model.get_input_embeddings().weight.device
        device_output = model.get_input_embeddings().weight.device
        with torch.no_grad():  # Ensure no gradient computation for these operations
            # Extract relevant data
            inputs_ids_convert = data['inputs_ids_convert.pkl']
            if emb_method!=RANDOM_EMB or CUSTOM_BPE:
                embedding_input = data[f'embedding_input_{emb_method}.pkl']
                embedding_output = data[f'embedding_output_{emb_method}.pkl']
            else:
                input_vocab_size, embedding_dim_input = model.get_input_embeddings().weight.shape
                output_vocab_size, embedding_dim_output = model.get_output_embeddings().weight.shape
                embedding_input = torch.randn((input_vocab_size, embedding_dim_input), device=device_input)
                embedding_output = torch.randn((output_vocab_size, embedding_dim_output), device=device_output)

            vocab_size = len(inputs_ids_convert)
            assert vocab_size == embedding_input.shape[0] == embedding_output.shape[0]
            new_input_embeddings = torch.nn.Embedding(num_embeddings=embedding_input.shape[0], embedding_dim=embedding_input.shape[1]).to(device_input)
            new_input_embeddings.weight = torch.nn.Parameter(torch.tensor(embedding_input, device=device_input, dtype=torch.float))
            new_output_embeddings = torch.nn.Linear(in_features=embedding_output.shape[0], out_features=embedding_output.shape[1], bias=False).to(device_output)
            new_output_embeddings.weight = torch.nn.Parameter(torch.tensor(embedding_output, device=device_output, dtype=torch.float))
            model.set_input_embeddings(new_input_embeddings)
            model.lm_head = new_output_embeddings

            return model
    
    if load_checkpoint:
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

        if emb_method!=VANILA_MODEL:
            tok_data=utils.load_data_from_directory(patch_tokenizer_path)
            model = modify_model(model,tok_data ,emb_method=emb_method)
        if use_lora:
            peft_config = LoraConfig( #TODO: fix the settings here to change the embeddigns
                task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, lora_dropout=0.1,
#                target_modules=["embed_tokens",'layers','mlp','input_layernorm','input_layernorm','norm','lm_head']
            )
            model = get_peft_model(model, peft_config)
        else:
            model = model.to(device)
            exact_unfrozen=True # not including first, just literly the num of layer
            for param in model.parameters():
                param.requires_grad = False
            for param in model.lm_head.parameters():
                param.requires_grad = True
            for param in model.get_input_embeddings().parameters():
                param.requires_grad = True
            
            end_layer = len(model.model.layers)
            start_layer = end_layer - unfroz[1]
            if unfroz[0] > 0:
                assert unfroz[0] <= len(model.model.layers), f"Unfrozen layer index {unfroz[0]} is out of range"
                utils.log_w_wandb(logger=logger, message=f"Unfrozen the first {unfroz[0]} layers")
                for layer_index in range(start_layer, unfroz[0]):
                    utils.log_w_wandb(logger=logger, message=f"Unfrozen layer: {layer_index}")
                    layer_to_unfroze = model.model.layers[layer_index]
                    for param in layer_to_unfroze.parameters():
                        param.requires_grad = True
            if unfroz[1] > 0:
                assert unfroz[1] <= len(model.model.layers), f"Unfrozen layer index {unfroz[1]} is out of range"
                end_layer = len(model.model.layers)
                start_layer = end_layer - unfroz[1]
                for layer_index in range(start_layer, end_layer):
                    utils.log_w_wandb(logger=logger, message=f"Unfrozen layer: {layer_index}")
                    layer_to_unfroze = model.model.layers[layer_index]
                    for param in layer_to_unfroze.parameters():
                        param.requires_grad = True

            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(name)  # This prints the names of parameters that are trainable
                    utils.log_w_wandb(logger=logger, message=f"Trainable parameter: {name}")

            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(name)  # This prints the names of parameters that are trainable
                    utils.log_w_wandb(logger=logger, message=f"Trainable parameter: {name}")
    
           # if emb_method!=VANILA_MODEL:
           #     for tok,tok_val in inputs_ids_convert.items():
           #         if type(tok_val['old_indices'])==int: # TODO: make sure it dosent harm preformence, maybe try without in future
           #             model.get_input_embeddings().weight[tok_val['new_index']].requires_grad = False
           #             model.lm_head.weight[tok['new_index']].requires_grad = False
    
    return model


def load_custom_model_kd(emb_method:str,model_to_load:str,patch_tokenizer_path:str=None
                      ,unfroz:Tuple[int,int]=(-1,-1)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def modify_model(model:AutoModelForCausalLM,inputs_ids_convert,emb_method:str) -> Tuple[AutoModelForCausalLM,Dict[str,Dict[str, Union[List[int], int]]],Dict[Tuple[str,...],float],List[str]]:
        """
        Modify the model using the loaded data.

        Parameters:
        model: The model to be modified.
        data: The data used for modifying the model, containing embeddings and token mappings.

        Returns:
        The modified model.
        """
        device_input=model.get_input_embeddings().weight.device
        device_output=model.lm_head.weight.device
        with torch.no_grad():  # Ensure no gradient computation for these operations
            org_input_embeddings = model.get_input_embeddings().weight
            org_lm_head = model.lm_head.weight
            if emb_method==EXP_EMB:
                new_input_embeddings_weights = change_embeddings_in_model.create_new_embbeding_matrix(org_input_embeddings,inputs_ids_convert,emb_method_str=INCREASING_EXP)
                new_output_embeddings_weights = change_embeddings_in_model.create_new_embbeding_matrix(org_lm_head,inputs_ids_convert,emb_method_str=DECREASING_EXP)
            elif emb_method==MEAN_EMB:
                new_input_embeddings = change_embeddings_in_model.create_new_embbeding_matrix(org_input_embeddings,inputs_ids_convert,emb_method_str=MEAN_EMB)
                new_output_embeddings = change_embeddings_in_model.create_new_embbeding_matrix(org_lm_head,inputs_ids_convert,emb_method_str=MEAN_EMB)
            else:
                raise NotImplementedError
        
            new_input_embeddings = torch.nn.Embedding(num_embeddings=new_input_embeddings_weights.shape[0], embedding_dim=new_input_embeddings_weights.shape[1]).to(device_input)
            new_input_embeddings.weight = torch.nn.Parameter(torch.tensor(new_input_embeddings_weights, device=device_input, dtype=torch.float))
            new_output_embeddings = torch.nn.Linear(in_features=new_output_embeddings_weights.shape[0], out_features=new_output_embeddings_weights.shape[1], bias=False).to(device_output)
            new_output_embeddings.weight = torch.nn.Parameter(torch.tensor(new_output_embeddings_weights, device=device_output, dtype=torch.float))
            model.set_input_embeddings(new_input_embeddings)
            model.lm_head = new_output_embeddings

            return model

    model = AutoModelForCausalLM.from_pretrained(model_to_load)
    if emb_method!=VANILA_MODEL:
        tok_data=utils.load_data_from_directory(patch_tokenizer_path)
        inputs_ids_convert = tok_data['inputs_ids_convert.pkl']
        model = modify_model(model,inputs_ids_convert ,emb_method=emb_method)
        
    #unfroze:
    
    model = model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.lm_head.parameters():
        param.requires_grad = True
    for param in model.get_input_embeddings().parameters():
        param.requires_grad = True
    
    if unfroz[0] > 0:
        assert unfroz[0] <= len(model.model.layers), f"Unfrozen layer index {unfroz[0]} is out of range"
        for layer_index in range(unfroz[0]):
            utils.log_w_wandb(logger=logger, message=f"Unfrozen layer: {layer_index}")
            layer_to_unfroze = model.model.layers[layer_index]
            for param in layer_to_unfroze.parameters():
                param.requires_grad = True

    if unfroz[1] > 0:
        assert unfroz[1] <= len(model.model.layers), f"Unfrozen layer index {unfroz[1]} is out of range"
        end_layer = len(model.model.layers)
        start_layer = end_layer - unfroz[1]
        for layer_index in range(start_layer, end_layer):
            utils.log_w_wandb(logger=logger, message=f"Unfrozen layer: {layer_index}")
            layer_to_unfroze = model.model.layers[layer_index]
            for param in layer_to_unfroze.parameters():
                param.requires_grad = True

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)  # This prints the names of parameters that are trainable
            utils.log_w_wandb(logger=logger, message=f"Trainable parameter: {name}")
                            
    return model

def get_model_path_name_from_hash(base_path:str,model_hash: str) -> str:
    for model_name in os.listdir(base_path):
        if model_name.startswith(model_hash):
            return os.path.join(base_path, model_name)
    return "Model not found."

def load_trained_mod_model(model_hash:str,spesific_checkpoint_path:str='None')->Tuple[AutoModelForCausalLM,PatchTokenizer,str]:
    hashes = utils.load_hash_mapping()
    utils.log_w_wandb(logger=logger, message=f"Loading model with hash {model_hash}")
    assert model_hash in hashes, f"Model hash {model_hash} not found in hash mapping"
    model_name=hashes[model_hash]
    utils.log_w_wandb(logger=logger, message=f"Loading model {model_name} with hash {model_hash}")
    if 'mistralai_Mistral-7B-Instruct-v0.2' in model_name:
        existing_tokenizer_name='mistralai/Mistral-7B-Instruct-v0.2'
    elif 'mistralai_Mistral-7B-v0.1' in model_name:
        existing_tokenizer_name='mistralai/Mistral-7B-v0.1'
    else:
        raise ValueError("Existing tokenizer couldn't be found")

    tokenizer_dir_path = utils.model_name_to_tokenizer_path(model_name)

    model_hash = get_model_path_name_from_hash(TRAINED_MODELS_PATH, model_hash)
    if spesific_checkpoint_path=='None':
        model_path=os.path.join(TRAINED_MODELS_PATH,model_hash)
        # check if model exists:
        utils.log_w_wandb(logger=logger, message=f"loading model from {model_path}")
        if not os.path.exists(model_path):
            raise ValueError(f"Model path {model_path} does not exist")
        checkpoint_path = utils.find_latest_checkpoint(model_path)
        utils.log_w_wandb(logger=logger, message=f"loading model from {checkpoint_path}")
    else:
        utils.log_w_wandb(logger=logger, message=f"loading model from spesific checkpoint path: {spesific_checkpoint_path}")
        checkpoint_path = spesific_checkpoint_path
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path).to('cuda')

    if '_vanila_' in model_name:
        utils.log_w_wandb(logger=logger, message=f"$$$$$ Vanila model {model_name} loaded")
        tokenizer = AutoTokenizer.from_pretrained(existing_tokenizer_name)
        
    elif os.path.exists(tokenizer_dir_path):
        print(f"Tokenizer exists in {tokenizer_dir_path}")
        #load:
        tokenizer = PatchTokenizer.load_model_from_scratch(path = os.path.join(tokenizer_dir_path,'patch_tokenizer.pkl') ,
                                                                    existing_tokenizer_name=existing_tokenizer_name)
    else:
        raise ValueError(f"Tokenizer does not exist in {tokenizer_dir_path} \n ___ \n Please create the patch tokenizer first.")

    return model, tokenizer,tokenizer_dir_path
    


class Evaluator:
    def __init__(self, model:AutoModelForCausalLM, tokenizer, dataset):
        # needs to get tokenized dataset
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset

    def evaluate(self):
        with torch.no_grad():  # Prevents tracking of gradients to reduce memory usage
            self.model.eval
            scaled_loss_sum = 0
            loss_sum = 0
            for text in self.dataset:
                input_ids =  torch.tensor(text['input_ids']).unsqueeze(0).to('cuda')
                loss = self.model(input_ids, labels=input_ids).loss
                scaled_loss = loss.detach().clone()  * len(input_ids[0])
                loss_sum += loss
                scaled_loss_sum += scaled_loss
                print(f"Loss: {loss}, Scaled Loss: {scaled_loss}")
                # Delete tensors explicitly to free up GPU memory
                del input_ids, loss, scaled_loss
            loss_sum = loss_sum / len(self.dataset)
            scaled_loss_sum = scaled_loss_sum / len(self.dataset)
            return scaled_loss_sum,loss_sum
        
    def generate_based_on_ds(self,is_patch_tokenizer:bool, max_length:int=100,model_nickname:str=None,exampels_to_generate:int=500,generations_folder:str=None,additioanl_name:str=''):
        model_nickname = model_nickname.split('/')[-1] if  model_nickname.split('/')[-1] else model_nickname.replace('/','_')
        model_nickname_hashed = utils.hash_output_dir(model_nickname)
        if generations_folder:
            csv_file_path = f'{generations_folder}/gens_{model_nickname_hashed}_{additioanl_name}.csv'
        else:
            csv_file_path = f'./generation_output_vanila_models/{model_nickname_hashed}_{additioanl_name}.csv'
        os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
        num_of_exampels = 0
        total_len=0
        ngrams_len =0
        c_ngrams_len=0
        ngrams_pres=0
        ngram_saving=0
        number_of_ngrams=0
        ngram_hist={}
        curr_ngram_hist={}
        with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if is_patch_tokenizer:
                writer.writerow(['Model Nickname', 'Ngrams', 'Tokens', 'Full Sentence', 'Sentence + Generated', 'Generated Sequence', 'Ngrams Presence','Ngrams_saving','Curr_ngram_saving'])
            else:
                writer.writerow(['Model Nickname', 'Tokens', 'Full Sentence', 'Sentence + Generated', 'Generated Sequence'])
            utils.log_w_wandb(logger=logger, message=f"$$ Generation evaluation for model, ds size: {len(self.dataset)}")
            for i,example in enumerate(self.dataset):
                c_ngrams_len=0
                if num_of_exampels>=exampels_to_generate:
                    break
                if len(example['input_ids'])<MIN_LEN_INPUT:
                    continue
                seq_len= min(PROMPT_LEN_LIST[i],len(example['input_ids']))
                cur_input = example['input_ids'][:seq_len]
                #find first None in the input_ids:
                last_token_idx =len(example['input_ids'])
                cur_input= torch.tensor(cur_input)
                cur_input=cur_input.to(self.model.device)
                cur_input=cur_input.unsqueeze(0)
                pad_token_id = self.tokenizer.existing_tokenizer.pad_token_id if is_patch_tokenizer else self.tokenizer.pad_token_id
                output_sequences = self.model.generate(
                input_ids=cur_input,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                early_stopping=True,
                do_sample=True,
                num_beams=1, 
                temperature=1.0,
                max_length=max_length,
                pad_token_id=pad_token_id)
                
                if is_patch_tokenizer:
                    generated_sequence,curr_ngram_hist = self.tokenizer.decode(output_sequences[0][seq_len:], skip_special_tokens=True,count_ngrams=True)
                    ngram_hist = utils.combine_ngram_hist(ngram_hist,curr_ngram_hist)
                    tokens_list = self.tokenizer.convert_ids_to_tokens(output_sequences[0][seq_len:])
                    total_len+=len(tokens_list)
                    c_number_of_ngrams = sum(curr_ngram_hist.values())
                    number_of_ngrams += c_number_of_ngrams
                    ngrams_pres = number_of_ngrams/total_len
                    for ng in curr_ngram_hist:
                        c_ngrams_len +=(curr_ngram_hist[ng])*len(ng)
                    c_ngram_saving= c_ngrams_len/(len(tokens_list)+c_ngrams_len-c_number_of_ngrams)
                    ngrams_len+=c_ngrams_len
                    ngram_saving= ngrams_len/(total_len+ngrams_len-number_of_ngrams)
                        # save to file, create file if not exists:
                    writer.writerow([
                        model_nickname,
                        str(curr_ngram_hist),
                        str(tokens_list),
                        self.tokenizer.decode(cur_input[0][:seq_len], skip_special_tokens=True) + " |||||| " + self.tokenizer.decode(torch.tensor(example['input_ids'][seq_len:last_token_idx]), skip_special_tokens=True),
                        self.tokenizer.decode(cur_input[0][:seq_len], skip_special_tokens=True) + " |||||| " + generated_sequence,
                        generated_sequence,
                        ngrams_pres,
                        ngram_saving,
                        c_ngram_saving
                    ])                        
                    logger.info(f'saved generated sequence to file in path {csv_file_path}')
                    num_of_exampels+=1
                       
                else:
                    tokens_list = self.tokenizer.convert_ids_to_tokens(output_sequences[0][seq_len:].tolist())
                    generated_sequence = self.tokenizer.decode(output_sequences[0][seq_len:], skip_special_tokens=True)
                    writer.writerow([
                        model_nickname,
                        str(tokens_list),
                        self.tokenizer.decode(cur_input[0][:seq_len], skip_special_tokens=True) + " |||||| " + self.tokenizer.decode(torch.tensor(example['input_ids'][seq_len:last_token_idx]), skip_special_tokens=True),
                        self.tokenizer.decode(cur_input[0][:seq_len], skip_special_tokens=True) + " |||||| " + generated_sequence,
                        generated_sequence,
                    ])   
                print(f'full sentence: {self.tokenizer.decode(cur_input[0][:seq_len], skip_special_tokens=True)} |||||| {self.tokenizer.decode(torch.tensor(example["input_ids"][seq_len:last_token_idx]), skip_special_tokens=True)}')
                print(f'sentence + generated: {self.tokenizer.decode(cur_input[0][:seq_len], skip_special_tokens=True)} |||||| {generated_sequence}')
                print(f'generated sequence: {generated_sequence}')
                print(f' ---------------------------------------------------------------------------------------- ')
                
                                     
                logger.info(f'saved generated sequence to file in path {csv_file_path}')
                num_of_exampels+=1
                
        return total_len,ngram_hist

def load_and_generate(original_model:str,is_patch: bool = False, model_hash: str = '', tokenizer_folder: str = '', cfg: Dict = None):
    # TODO: update this func 
    if is_patch:
        if model_hash:
            # load modifiy and trained model
            model, tokenizer = load_trained_mod_model(model_hash)
            hashes = utils.load_hash_mapping()
            model_name=hashes[model_hash]
            assert 'Culture' in model_name or 'Natural' in model_name, "Model name does not contain known target corpus"
            if 'Culture_and_the_arts__The_arts_and_Entertainment' in model_name:
                path_to_tokenized_ds = f'{REPO_PATH}/old/saved_patch_tokenizers_old1/original_tokenizer_mistralai_Mistral-7B-Instruct-v0.2_target_corpus_name_machelreid_m2d2_Culture_and_the_arts__The_arts_and_Entertainment_num_to_add_10000_num_to_remove_10000_ngram_k_4_/tokenized_datasets'
                target_courpus_name = 'machelreid/m2d2_Culture_and_the_arts__The_arts_and_Entertainment'
            elif 'Natural_and_physical_sciences__Earth_sciences' in model_name:
                path_to_tokenized_ds = f'{REPO_PATH}/old/saved_patch_tokenizers_old1/original_tokenizer_mistralai_Mistral-7B-Instruct-v0.2_target_corpus_name_machelreid_m2d2_Natural_and_physical_sciences__Earth_sciences_num_to_add_10000_num_to_remove_10000_ngram_k_4_/tokenized_datasets'
                target_courpus_name = 'machelreid/m2d2_Natural_and_physical_sciences__Earth_sciences'
            if '_vanila_' in model_name:
                is_patch=False
            model_nick=model_hash
        else:
            # load modifiy and untrained model
            assert tokenizer_folder, "Please provide a path to the patch tokenizer"
            path_to_tokenized_ds = f'{tokenizer_folder}/tokenized_datasets'
            assert os.path.exists(path_to_tokenized_ds), f"Path to tokenized datasets {path_to_tokenized_ds} does not exist"
            cfg = pickle.load(open(f'{tokenizer_folder}/config.pkl', 'rb'))
            tokenizer = PatchTokenizer.load_model_from_scratch(
                path=f'{tokenizer_folder}/patch_tokenizer.pkl',
                existing_tokenizer_name=original_model
            )
            model = load_custom_model(
                model_name=original_model,
                emb_method=cfg['emb_method'],
                patch_tokenizer_path=tokenizer_folder,
                load_checkpoint=False
            )
            target_courpus_name = cfg['target_corpus_name']
            model_nick='_untrained_'+utils.cfg_to_filename(cfg)
    else:
        # load untrained and unmodify model
        assert cfg, "Please provide a config"
        model = AutoModelForCausalLM.from_pretrained(original_model).to('cuda')
        tokenizer = AutoTokenizer.from_pretrained(original_model)
        path_to_tokenized_ds = f'{REPO_PATH}/saved_patch_tokenizers/_vanila_mistralai/Mistral-7B-v0.1_Natural_and_physical_sciences__Earth_sciences'
        assert os.path.exists(path_to_tokenized_ds), f"Path to tokenized datasets {path_to_tokenized_ds} does not exist"
        target_courpus_name = cfg['target_corpus_name']
        model_nick = '_untrained_unmodified_'+utils.cfg_to_filename(cfg)

    ds_name, ds_config = utils.get_ds_config_names(target_courpus_name)
    dp = dp_train.DataPrepare(dataset_name=ds_name, tokenizer=tokenizer, dataset_config=ds_config)
    tokenized_ds = dp.load_and_prepare_data(0, 10_000, tokenized_data_path=path_to_tokenized_ds)
    evaluator = Evaluator(model=model, tokenizer=tokenizer, dataset=tokenized_ds['validation'])
    evaluator.generate_based_on_ds(is_patch_tokenizer=is_patch, model_nickname=model_nick)

    
def generate_vanila_model(original_model:str='mistralai/Mistral-7B-v0.1'):
    cfg={'analyze_ds_names':['machelreid/m2d2_Natural_and_physical_sciences__Earth_sciences']}
    model = AutoModelForCausalLM.from_pretrained(original_model).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(original_model)
    ds_name,ds_config= utils.get_ds_config_names(cfg['target_corpus_name'])
    dp = dp_train.DataPrepare(dataset_name=ds_name[0], tokenizer=tokenizer,dataset_config=ds_config[0])
    path_to_tokenized_ds=os.path.join(SAVED_PATCH_TOKENIZER_PATH, VANILA_TOKENIZER_NAME)
    tokenized_ds= dp.load_and_prepare_data(0 ,5_000,tokenized_data_path=path_to_tokenized_ds)
    evaluator = Evaluator(model=model,tokenizer=tokenizer,dataset=tokenized_ds['validation'])
    evaluator.generate_based_on_ds(is_patch_tokenizer=False,model_nickname='exp_initialize_physical')

def generate_untrained_mod_model(cfg:Dict,tokenizer_folder:str):
    original_model = cfg['original_tokenizer']
    path_to_tokenized_ds=f'{tokenizer_folder}/tokenized_datasets'
    p_tokenizer=PatchTokenizer.load_model_from_scratch(path = tokenizer_folder+'/patch_tokenizer.pkl' ,existing_tokenizer_name=original_model)
    model = load_custom_model(model_name=original_model,emb_method=cfg['emb_method'],patch_tokenizer_path=tokenizer_folder,use_lora=False)
    ds_name,ds_config= utils.get_ds_config_names(cfg['target_corpus_name'])
    dp = dp_train.DataPrepare(dataset_name=ds_name, tokenizer=p_tokenizer,dataset_config=ds_config)
    tokenized_ds= dp.load_and_prepare_data(0 ,10_000,tokenized_data_path=path_to_tokenized_ds)
    evaluator = Evaluator(model=model,tokenizer=p_tokenizer,dataset=tokenized_ds['validation'])
    generations_folder='/data/home/itay.nakash/patch_tokenizer/generation_untrained_models/'
    evaluator.generate_based_on_ds(is_patch_tokenizer=cfg['emb_method']!=VANILA_MODEL,model_nickname=utils.cfg_to_filename(cfg),exampels_to_generate=251,generations_folder=generations_folder)

def load_mod_model_and_generate(model_hash:str,is_patch: bool,spesific_checkpoint_paths:List[str]=[]):
    add_name=''
    if spesific_checkpoint_paths == []:
        model,tokenizer,tokenizer_dir_path=load_trained_mod_model(model_hash)
    else:
        for cp_path in spesific_checkpoint_paths:
            utils.log_w_wandb(logger=logger, message=f"$$$$$$$$$$$ Loading model with cp path {cp_path}")
            utils.log_w_wandb(logger=logger, message=f"$$$$$$$$$$$ free cuda mem: {torch.cuda.memory_allocated()}")
            model,tokenizer,tokenizer_dir_path=load_trained_mod_model(model_hash,cp_path)
            checkpoint_num = int(re.search(r"checkpoint-(\d+)", cp_path).group(1))
            add_name=checkpoint_num
            
            
            hashes = utils.load_hash_mapping()
            model_name=hashes[model_hash]
            assert 'Culture' in model_name or 'Natural' in model_name, "Model name does not contain known target corpus"
            if 'Culture_and_the_arts__The_arts_and_Entertainment' in model_name:
                path_to_tokenized_ds = f'{tokenizer_dir_path}/tokenized_datasets' if 'vanila' not in model_name else tokenizer_dir_path
                target_courpus_name = 'machelreid/m2d2_Culture_and_the_arts__The_arts_and_Entertainment'
            elif 'Natural_and_physical_sciences__Earth_sciences' in model_name:
                path_to_tokenized_ds = f'{tokenizer_dir_path}/tokenized_datasets' if 'vanila' not in model_name else tokenizer_dir_path
                target_courpus_name = 'machelreid/m2d2_Natural_and_physical_sciences__Earth_sciences'
            ds_name, ds_config = utils.get_ds_config_names(target_courpus_name)
            dp = dp_train.DataPrepare(dataset_name=ds_name, tokenizer=tokenizer, dataset_config=ds_config)
            tokenized_ds = dp.load_and_prepare_data(0, 10_000, tokenized_data_path=path_to_tokenized_ds)
            evaluator = Evaluator(model=model, tokenizer=tokenizer, dataset=tokenized_ds['validation'])
            evaluator.generate_based_on_ds(is_patch_tokenizer=is_patch, model_nickname=model_hash,additioanl_name=add_name,generations_folder='/data/home/itay.nakash/patch_tokenizer/outputs/generation_outputs/')
            #free all cuda mem used:
            del evaluator
            del dp
            del tokenized_ds
            del model
            del tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()
            device = torch.device('cuda:0')  # or the appropriate device
            torch.cuda.set_device(device)
            torch.cuda.empty_cache()

            print(f'finished generating for checkpoint {add_name}')
            #cuda memory check:
            print(f'cuda memory allocated: {torch.cuda.memory_allocated()}')    

def load_mod_model_and_qa(model_hash:str,is_patch: bool,spesific_checkpoint_paths:List[str]=[]):
    add_name=''
    if spesific_checkpoint_paths == []:
        model,tokenizer,tokenizer_dir_path=load_trained_mod_model(model_hash)
    else:
        for cp_path in spesific_checkpoint_paths:
            utils.log_w_wandb(logger=logger, message=f"$$$$$$$$$$$ Loading model with cp path {cp_path}")
            utils.log_w_wandb(logger=logger, message=f"$$$$$$$$$$$ free cuda mem: {torch.cuda.memory_allocated()}")
            model,tokenizer,tokenizer_dir_path=load_trained_mod_model(model_hash,cp_path)
            checkpoint_num = int(re.search(r"checkpoint-(\d+)", cp_path).group(1))
            add_name=checkpoint_num
            hashes = utils.load_hash_mapping()
            model_name=hashes[model_hash]
            model_nick=f'{model_hash}_checkpoint:{checkpoint_num}'
            evaluate.QA.eval.eval_on_all_tasks(model, tokenizer,model_nick)
            del model
            del tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()
            device = torch.device('cuda:0')  # or the appropriate device
            torch.cuda.set_device(device)
            torch.cuda.empty_cache()

            print(f'finished generating for checkpoint {add_name}')
            #cuda memory check:
            print(f'cuda memory allocated: {torch.cuda.memory_allocated()}')    

    
if __name__ == "__main__":
   cfg_tok=CONFIG_P_TOKENIZER
   cfg_ft=CONFIG_FT
   cfg = {**cfg_tok, **cfg_ft}
   #test untrained:
  # tokenizer_folder='/data/home/itay.nakash/patch_tokenizer/saved_patch_tokenizers/original_tokenizer_mistralai_Mistral-7B-v0.1_target_corpus_name_machelreid_m2d2_Natural_and_physical_sciences__Earth_sciences_num_to_add_10000_num_to_remove_10000_ngram_k_4_date_major_up_0306_'
  # original_model='mistralai/Mistral-7B-v0.1'
  # generate_untrained_mod_model(cfg,tokenizer_folder)
   
   #get model_hash and checkpoint_path base from args:
   if len(sys.argv)>1:
        model_hash=sys.argv[1]
        checkpoints_base_path=sys.argv[2]
        #get all folders names in the checkpoints_base_path:
        checkpoint_paths = [f'{checkpoints_base_path}/{folder}' for folder in os.listdir(checkpoints_base_path) if os.path.isdir(f'{checkpoints_base_path}/{folder}')]
        load_mod_model_and_generate(model_hash=model_hash,is_patch=True,spesific_checkpoint_paths=checkpoint_paths)
   #tokenizer_folders=[f'{REPO_PATH}/saved_patch_tokenizers/original_tokenizer_mistralai_Mistral-7B-Instruct-v0.2_big_corpus_name_None_target_corpus_name_machelreid_m2d2_Culture_and_the_arts__The_arts_and_Entertainment_num_to_add_10000_num_to_remove_10000_alpha_0_ngram_k_4_emb_method_exp_emb_',f'{REPO_PATH}/saved_patch_tokenizers/original_tokenizer_mistralai_Mistral-7B-Instruct-v0.2_big_corpus_name_None_target_corpus_name_machelreid_m2d2_Culture_and_the_arts__The_arts_and_Entertainment_num_to_add_10000_num_to_remove_10000_alpha_0_ngram_k_4_emb_method_mean_emb_',f'{REPO_PATH}/saved_patch_tokenizers/original_tokenizer_mistralai_Mistral-7B-Instruct-v0.2_big_corpus_name_None_target_corpus_name_machelreid_m2d2_Natural_and_physical_sciences__Earth_sciences_num_to_add_10000_num_to_remove_10000_alpha_0_ngram_k_4_emb_method_exp_emb_',f'{REPO_PATH}/saved_patch_tokenizers/original_tokenizer_mistralai_Mistral-7B-Instruct-v0.2_big_corpus_name_None_target_corpus_name_machelreid_m2d2_Natural_and_physical_sciences__Earth_sciences_num_to_add_10000_num_to_remove_10000_alpha_0_ngram_k_4_emb_method_mean_emb_']
    #set cuda to 0:
    #vanila model trained:
    #checkpoint_paths=[f'{REPO_PATH}/trained_models_0506/model_7cfcc001c1_sweep_usual-puddle-578/checkpoint-{i}' for i in range(100,3600,100) ]
    #load_mod_model_and_generate(model_hash='model_7cfcc001c1',is_patch=False,spesific_checkpoint_paths=checkpoint_paths)
    #AV exp trained:
    #checkpoint_paths=[f'{REPO_PATH}/trained_models_0506/model_3d7f448ce8_sweep_dauntless-hill-577/checkpoint-{i}' for i in range(100,3600,100)]
    #load_mod_model_and_qa(model_hash='model_3d7f448ce8',is_patch=True,spesific_checkpoint_paths=checkpoint_paths)
    # model_90598d0146_sweep_fallen-jazz-575
    
    