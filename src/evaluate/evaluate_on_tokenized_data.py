import sys
import os
sys.path.append((os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import gc
import torch
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from pathlib import Path
from build_vocab import load_custom_model 
from utils.constants import *
from build_vocab.patch_tokenizer import PatchTokenizer
import utils.utils_funcs as utils
import pickle
import evaluate.evaluate_model_outputs
import numpy as np


def process_inputs(input_ids:List[int],model,tokenizer)->Tuple[str,Dict[str,int]]:
    tensor_input = torch.tensor(input_ids).cuda()
    if len(tensor_input.shape) == 1:
        tensor_input = tensor_input.unsqueeze(0)
    max_length=200+len(tensor_input[0])
    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    with torch.no_grad():
        output = model.generate(
                input_ids=tensor_input,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                early_stopping=True,
                do_sample=True,
                num_beams=1, 
                temperature=1.0,
                max_length=max_length,
                pad_token_id=pad_token_id)
    if type(tokenizer)==PatchTokenizer:
        output,ngrams_hist = tokenizer.decode(output[0], skip_special_tokens=True,count_ngrams=True)
    else:
        output = tokenizer.decode(output[0], skip_special_tokens=True)
        ngrams_hist={}
    return output,ngrams_hist

def _evaluate_preplexity(dataset, model):
    total_loss=0
    count=0
    with torch.no_grad():
        for example in tqdm(dataset):
            input_ids = torch.tensor(example['prompt']).unsqueeze(0).to(model.device)
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            total_loss += loss.item()
            count += 1

    mean_loss = total_loss / count
    perplexity = torch.exp(torch.tensor(mean_loss)).item()

    return perplexity


def evaluate_and_save_he(path:str, tokenizer, global_step: int,model:AutoModelForCausalLM,
                      output_path:str,max_length:int=50):
        with open(path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        # Access inputs directly
        inputs = dataset['inputs']
        outputs = {'outputs': [],'trim_outputs':[], 'outputs_as_tokens':[], 'correct_answer': [], 'inputs': []}
                        
        for i,input_ids in enumerate(inputs):
            max_length = max_length
            input_ids = torch.tensor(input_ids).to(model.device)
            input_ids = input_ids.unsqueeze(0)
            output_ids = model.generate(
            input_ids=input_ids,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            early_stopping=True,
            do_sample=True,
            num_beams=1, 
            temperature=1.0,
            max_length=max_length+len(input_ids[0]),
            pad_token_id=tokenizer.pad_token_id)
            decoded_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            decoded_tokens = tokenizer.convert_ids_to_tokens(output_ids[0])
            #if len(outputs['outputs'])<5:
                #assert correct tokenizer:
                #assert self.tokenizer.decode(sample['input_ids'][0])==sample['prompt'], f"decoded: {self.tokenizer.decode(sample['input_ids'][0])} prompt: {sample['prompt']}"
            outputs['outputs'].append(decoded_output)
            outputs['trim_outputs'].append(decoded_output[len(dataset['prompts'][i]):])
            outputs['outputs_as_tokens'].append(decoded_tokens)
            outputs['inputs'].append(tokenizer.decode(input_ids[0]))

        # Save the outputs
        # current_output path:
        c_out_p = output_path+f"_step_{global_step}.json"
        utils.log_w_wandb(logger=logger, message=f"----- Saving outputs to {c_out_p}")
        with open(c_out_p, 'w') as f:
            json.dump(outputs, f, ensure_ascii=False, indent=4)


def evaluate_and_save(eval_datasets:Dict[str,str], tokenizer, global_step: int,model:AutoModelForCausalLM,
                      output_paths:Dict[str,str],max_length:int=50,number_of_qa_to_eval:int=20):
    for dataset_name, dataset_path in eval_datasets.items():
        # TODO: find a more elegant way (define a list somehow):
        # TODO: add preplexity:
        # and 'perplexity_data' not in dataset_name
        
        #datasets_to_eval_on = ['domain_qa_context', 'domain_qa_context_over_train', 'generated_data','perplexity_data']
        #just for physical:
        #datasets_to_eval_on = ['generated_data']
        datasets_to_eval_on = ['generated_data','domain_qa_context', 'domain_qa_context_over_train']
        
        #datasets_to_eval_on = ['domain_qa_context','domain_qa_context_over_train', 'generated_data','perplexity_data','domain_qa_no_context_over_train']
        #datasets_to_eval_on = ['generated_data','perplexity_data']
        #datasets_to_eval_on = ['perplexity_data']
        if not any(ds in dataset_name for ds in datasets_to_eval_on):
            continue
        #load dataset from json:
        dataset=[]
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        outputs = {'outputs': [],'trim_outputs':[], 'outputs_as_tokens':[], 'correct_answer': [], 'inputs': []}
        
        if 'perplexity_data' in dataset_path:
                # evaluate prep
                preplex=_evaluate_preplexity(dataset, model)
                #save it:
                c_out_p = output_paths[dataset_name]+f"_step_{global_step}.json"
                with open(c_out_p, 'w') as f:
                    json.dump({'perplexity':preplex}, f, ensure_ascii=False, indent=4)
                utils.log_w_wandb(logger=logger, message=f"----- Saving {dataset_name} outputs to {c_out_p}")
                continue
                
        for sample in dataset:
            if 'qasc_qa' in dataset_path or 'ecqa_qa' in dataset_path or 'arc_qa' in dataset_path or 'openbookqa_qa' in dataset_path:
                if len(outputs['outputs'])>=number_of_qa_to_eval:
                    break
                
            if utils.is_debug_mode() and len(outputs['outputs'])>=5:
                break
            if len(outputs['outputs'])>=200:
                break
            input_ids = sample['input_ids']
            max_length = max_length if 'generated_data' not in dataset_path else 50 # generate 50 max if not eval 
            input_ids = torch.tensor(input_ids).to(model.device)
            input_ids = input_ids.unsqueeze(0)
            output_ids = model.generate(
            input_ids=input_ids,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            early_stopping=True,
            do_sample=True,
            num_beams=1, 
            temperature=1.0,
            max_length=max_length+len(input_ids[0]),
            pad_token_id=tokenizer.pad_token_id)
            decoded_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            decoded_tokens = tokenizer.convert_ids_to_tokens(output_ids[0])
            #if len(outputs['outputs'])<5:
                #assert correct tokenizer:
                #assert self.tokenizer.decode(sample['input_ids'][0])==sample['prompt'], f"decoded: {self.tokenizer.decode(sample['input_ids'][0])} prompt: {sample['prompt']}"
            outputs['outputs'].append(decoded_output)
            outputs['trim_outputs'].append(decoded_output[len(sample['prompt']):])
            outputs['outputs_as_tokens'].append(decoded_tokens)
            outputs['correct_answer'].append(sample['correct_answer'])
            outputs['inputs'].append(tokenizer.decode(input_ids[0]))

        # Save the outputs
        # current_output path:
        c_out_p = output_paths[dataset_name]+f"_step_{global_step}.json"
        utils.log_w_wandb(logger=logger, message=f"----- Saving {dataset_name} outputs to {c_out_p}")
        with open(c_out_p, 'w') as f:
            json.dump(outputs, f, ensure_ascii=False, indent=4)



# !! FUNCTION that evaluate model on tokenized data !!
def main(model_name,checkpoint_list_exp,main_dir):
    for cp_p in checkpoint_list_exp:
        model,tokenizer,tok_path = load_custom_model.load_trained_mod_model(model_hash=model_name,spesific_checkpoint_path=cp_p)
        # Process files in the main directory
        eval_datasets = utils.get_eval_ds_path(tok_path)
        model_output_dir = f"{EVAL_OUTPUTS_PATH}/kd/{model_name}_outputs"
        os.makedirs(model_output_dir, exist_ok=True)
        output_paths = {name: f"{model_output_dir}/{name}_output.json" for name in eval_datasets}
        evaluate_and_save(eval_datasets=eval_datasets, tokenizer=tokenizer, global_step=cp_p.split('-')[-1],
                          model=model,output_paths=output_paths)
        evaluate.evaluate_model_outputs.evaluate_and_save_to_csv(output_dir=model_output_dir, csv_output_path=f"{model_output_dir}/summary_results.csv")
        # free memory
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_pkl_and_print(path:str):
    with open(path, 'rb') as file:
        outputs = pickle.load(file)
    for key in outputs:
        print(f"Key: {key}")
        for sub_key in outputs[key]:
            print(f"Sub-key: {sub_key}")
            if sub_key == 'outputs':
                for output in outputs[key][sub_key]:
                    print(output)
            elif sub_key == 'ngram_hist':
                print(outputs[key][sub_key])
            else:
                print(outputs[key][sub_key])
    return outputs
 
 
        
if __name__ == "__main__":

    model_name='model_3d7f448ce8'
    checkpoint_list_exp=[f'{REPO_PATH}/trained_models_0506/model_3d7f448ce8_sweep_dauntless-hill-577/checkpoint-{i}' for i in range(100,6200,100) ]
    main_dir=f"{REPO_PATH}/evaluate/tokenized_eval_data/eval_data/original_tokenizer_mistralai_Mistral-7B-v0.1_target_corpus_name_machelreid_m2d2_Natural_and_physical_sciences__Earth_sciences_num_to_add_10000_num_to_remove_10000_ngram_k_4_date_major_up_0306_"
    main(model_name=model_name,checkpoint_list_exp=checkpoint_list_exp,main_dir=main_dir)
    
