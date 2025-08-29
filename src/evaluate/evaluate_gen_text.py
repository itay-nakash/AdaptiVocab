import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import utils.utils_funcs as utils
import pandas as pd
from typing import Dict, List, Union,Tuple
from transformers import (
    AutoTokenizer, AutoModelForCausalLM
)
import os
from utils.constants import *
from transformers import pipeline
import json
import matplotlib.pyplot as plt
import datasets
from datasets import Dataset
from vertexai import generative_models


# utils:
def _read_gens_from_csv(file_path:str)->pd.DataFrame:
    """Read CSV file into a DataFrame and separate prompts from generated text.
    
    Args:
        file_path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: DataFrame with 'prompt' and 'generated_text' columns.
    """
    # Reading the CSV file using the specified separator
    df = pd.read_csv(file_path, engine='python')
    
    # Splitting 'Full Sentence' into 'prompt' and any trailing content, if present
    # Assuming 'Full Sentence' ends with '||' which signifies the end of the prompt
    df['prompt'] = df['Full Sentence'].apply(lambda x: str(x).split('||||||')[0])
    
    #check number of nan values in 'Generated Sequence' column:
    logger.log(level=logging.INFO,msg=f"Number of nan values in 'Generated Sequence' column: {df['Generated Sequence'].isnull().sum()}")
    
    # remove rows that has nan in 'Generated Sequence' column:
    df = df.dropna(subset=['Generated Sequence'])
    df['generated_text'] = df['Generated Sequence']
    df.drop(columns=['Full Sentence'], inplace=True)
    df.drop(columns=['Generated Sequence'], inplace=True)
    
    return df

def _get_dataset_as_prompt_gen_by_ref(ds: Dataset,ref_gens:pd.DataFrame) -> pd.DataFrame:
    ds_to_eval = pd.DataFrame(columns=['prompt','generated_text'])
    # merge ds['validation'] to 256 length strings:
    rows = []  # to accumulate rows for the final DataFrame
    flat_ds = ' '.join(ds['validation']['text'])
    flat_ds = flat_ds.split(' ')
    flat_ds = [row for row in flat_ds if row!='']
    c_text_added=0
    for i,_ in enumerate(ref_gens['prompt']):
        prompt_len = len(ref_gens['prompt'][i].split(' '))
        gen_len=len(ref_gens['generated_text'][i].split(' '))
      #  assert flat_ds[c_text_added:c_text_added+prompt_len-1]==ref_gens['prompt'][i]
        prompt=' '.join(flat_ds[c_text_added:(c_text_added)+prompt_len-1])
        generated_text=' '.join(flat_ds[c_text_added+prompt_len:(c_text_added+prompt_len)+gen_len])
        rows.append({'prompt': prompt, 'generated_text': generated_text})
        c_text_added+=256
    ds_to_eval = pd.DataFrame(rows, columns=['prompt', 'generated_text'])

    return ds_to_eval
def _get_dataset_as_prompt_gen(ds: Dataset,num_to_eval:int=200,text_len:int=100) -> pd.DataFrame:
    ds_to_eval = pd.DataFrame(columns=['prompt','generated_text'])
    # merge ds['validation'] to 256 length strings:
    rows = []  # to accumulate rows for the final DataFrame
    filter_ds_len = [row for row in ds['validation']['text'] if len(row.split(' '))>40]
    flat_ds = ' '.join(filter_ds_len)
    flat_ds = flat_ds.split(' ')
    flat_ds = [row for row in flat_ds]
    c_text_added=0
    for i in range(num_to_eval):
        prompt_len= PROMPT_LEN_LIST[i]
        assert prompt_len < text_len
        prompt= ' '.join(flat_ds[c_text_added: c_text_added+prompt_len])
        generated_text=' '.join(flat_ds[c_text_added+prompt_len: text_len])
        rows.append({'prompt': prompt, 'generated_text': generated_text})
        c_text_added+=256
    ds_to_eval = pd.DataFrame(rows, columns=['prompt', 'generated_text'])

    return ds_to_eval    

def _generate_with_model(model,evaluation_input:str,n_gen_toks:int,tokenizer=''):
    # Eval using gemini
    if type(model)==generative_models.GenerativeModel:
        utils.log_w_wandb(logger=logger, message=f"Using gemini model to generate text.",level='info')
        evaluation_input+=' [/INST] EVALUATION OUTPUT, ANSWER WITH NUMBER FIRST:'
        prediction = model.generate_content(evaluation_input)
        try:
            output =  prediction.candidates[0].content.parts[0]._raw_part.text
        except:
            print('no answer from gemini')
            output = 'NO ANSWER'
        output=' [/INST] ' + output
    # Eval using Mistral
    else:
        assert tokenizer!='', 'tokenizer must be provided'
        #eval using mistral:
        mistral_input_formated=[{"role":"user","content":evaluation_input}]
        model_inputs = tokenizer.apply_chat_template(mistral_input_formated, return_tensors="pt").to("cuda")
        generated_ids = model.generate(model_inputs, max_new_tokens=n_gen_toks, do_sample=True)
        output = tokenizer.batch_decode(generated_ids)[0]
        
    return output

def _evaluate_aspect(model,tokenizer:AutoTokenizer, aspect_prompt:str, data: pd.DataFrame)->List[int]:
    n_gen_toks=10
    num_of_tries=5
    results = []
    for _, row in data.iterrows():
        for _ in range(num_of_tries):
            evaluation_input = f"{aspect_prompt} {row['prompt']} {row['generated_text']}"
            output = _generate_with_model(model,evaluation_input,n_gen_toks,tokenizer)
            has_num,num = _get_num_from_gen(output)
            if has_num:
                results.append(num)
                break
    return results

def _get_num_from_gen(gen_text: str) -> Tuple[bool, int]:
    gen_text = gen_text.split(" [/INST]")[1]
    try:
        numbers = [int(c) for c in gen_text if c.isdigit()]
        if numbers:
            return True, numbers[0]
        else:
            return False, -1
    except:
        return False, -1

def evaluate_with_LLM(model, generated_text: pd.DataFrame, aspect_prompts: Dict[str, str], tokenizer='', num_examples: int = None) -> pd.DataFrame:
    """Evaluate the data using LLM model.

    Args:
        generated_text (pd.DataFrame): Data containing generated texts.
        num_examples (int): Number of examples to evaluate. If None, evaluate all.

    Returns:
        pd.DataFrame: Evaluation scores or relevant metrics including averages.
    """
    if num_examples is not None:
        generated_text = generated_text.head(num_examples)

    evaluation_results = {}
    for aspect, prompt in aspect_prompts.items():
        aspect_scores = _evaluate_aspect(model, tokenizer, prompt, generated_text)
        evaluation_results[aspect] = {
            'Scores': aspect_scores,
            'Average': sum(aspect_scores) / len(aspect_scores) if aspect_scores else 0,
            'Std': np.std(aspect_scores) if aspect_scores else 0
        }

    # Convert the dictionary to DataFrame correctly
    results_list = []
    for aspect, scores in evaluation_results.items():
        results_list.append({
            'Aspect': aspect,
            'Scores': scores['Scores'],
            'Average': scores['Average'],
            'Std': scores['Std']
        })
    return pd.DataFrame(results_list)

def _set_evaluator_model(evaluator_name:str)->AutoModelForCausalLM:
    evaluator_model = AutoModelForCausalLM.from_pretrained(evaluator_name).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(evaluator_name)
    tokenizer.pad_token = tokenizer.eos_token
    evaluator_model.config.pad_token_id = evaluator_model.config.eos_token_id
    return evaluator_model,tokenizer

def evaluate_gens_with_LLM(evaluator_name:str, aspect_prompts:dict, gen_dir_path:str, gens_name:str, llm_cfg: str)->pd.DataFrame:
    file_path = os.path.join(gen_dir_path, gens_name)
    evaluator_model,tokenizer = _set_evaluator_model(evaluator_name)
    data = _read_gens_from_csv(file_path)
    if utils.is_debug_mode():
        data = data.head(5)
    evaluation_results = evaluate_with_LLM(model=evaluator_model,generated_text= data,aspect_prompts= aspect_prompts,tokenizer= tokenizer)
    evaluation_results['LLM Config'] = llm_cfg 
    return evaluation_results

def plot_aspect_histograms(all_results: pd.DataFrame, gen_dir_path: str) -> None:
    grouped = all_results.groupby('Model Nickname')
    for model_hash, group in grouped:
        plt.figure(figsize=(10, 6))
        aspects = group['Aspect'].unique()
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'black', 'brown', 'grey', 'pink']  # Ensure enough colors
        for i, aspect in enumerate(aspects):
            aspect_data = group[group['Aspect'] == aspect]['Scores']
            aspect_data.value_counts().sort_index().plot(kind='bar', color=colors[i], alpha=0.7, position=i, width=0.1, label=aspect)

        plt.title(f'Scores Distribution for {model_hash}')
        plt.xlabel('Scores')
        plt.ylabel('Frequency')
        plt.xticks(rotation=0)
        plt.legend(title='Aspects')
        plt.savefig(os.path.join(gen_dir_path, f'{model_hash}_all_aspects_histogram.png'))
        plt.close()


def add_features_to_pd(results: pd.DataFrame) -> pd.DataFrame:
    def extract_dataset(LLM_config: str) -> str:
        # Mapping for dataset
        mapping = {
            "Culture_and_the_arts__The_arts_and_Entertainment": "Culture",
            "Natural_and_physical_sciences__Earth_sciences": "Physical"
        }
        for key, value in mapping.items():
            if key in LLM_config:
                return value
        if 'culture' in LLM_config:
            return 'vanilla_Culture'
        return NOT_FOUND

    def extract_method(LLM_config: str) -> str:
        # Determine method
        if "emb_method_exp" in LLM_config:
            return "exp"
        elif "mean" in LLM_config:
            return "mean"
        return NOT_FOUND

    def extract_lora(LLM_config: str) -> str:
        # Determine LORA usage
        if "use_lora_True" in LLM_config:
            return True
        elif "use_lora_False" in LLM_config:
            return False
        return False

    # Apply the functions to extract each configuration part
    results['Dataset'] = results['LLM Config'].apply(extract_dataset)
    results['Method'] = results['LLM Config'].apply(extract_method)
    results['Lora Status'] = results['LLM Config'].apply(extract_lora)
    return results
    

def main(gen_dir_path:str,seeds=[44]):
    evaluator_name = 'mistralai/Mistral-7B-Instruct-v0.2'
    hashes = json.load(open('hashes.json'))  # Load the hash mapping
    # if not exitst - create evaluation results directory:
    if not os.path.exists(f"{gen_dir_path}/evaluation_results"):
        os.makedirs(f"{gen_dir_path}/evaluation_results")
    
    eval_results_dir = f"{gen_dir_path}/evaluation_results"
    if not os.path.exists(eval_results_dir):
        os.makedirs(eval_results_dir)
    all_results = pd.DataFrame() 
    for seed in seeds:
        c_seed_results = pd.DataFrame() 
        c_path = os.path.join(eval_results_dir, f"seed_{seed}")
        if not os.path.exists(c_path):
            os.makedirs(c_path)
        utils.set_seed(SEED + seed)
        
        for gens_file in tqdm(os.listdir(gen_dir_path)):
            if gens_file.endswith(".csv"):
                model_hash = '_'.join(gens_file.split('_')[:2])
                model_checkpoint = gens_file.split('_')[-1].replace('.csv', '')
                llm_name = hashes.get(model_hash, 'default_model_name')
                results = evaluate_gens_with_LLM(evaluator_name, ASPECT_PROMPTS, gen_dir_path, gens_file, llm_name)
                results['Model Nickname'] = model_hash
                results['Checkpoint'] = model_checkpoint
                
                # For each model (hash), calculate the average score for all aspects:
                results = add_features_to_pd(results)
                results['avg_score'] = results[results['Model Nickname'] == model_hash]['Average'].mean()
                logger.log(level=logging.INFO, msg=f"Finished evaluating {model_hash} model on seed {seed}.")
                all_results = pd.concat([all_results, results], ignore_index=True)
                
                # Ensure each model in different row
                c_seed_results = pd.concat([c_seed_results, results], ignore_index=True)
                c_seed_results = pd.concat([c_seed_results, pd.DataFrame([{}])], ignore_index=True)
                
                # Update and save the results for the current generation file
                csv_path = os.path.join(c_path, "evaluation_results.csv")
                if os.path.exists(csv_path):
                    existing_results = pd.read_csv(csv_path)
                    combined_results = pd.concat([existing_results, results], ignore_index=True)
                else:
                    combined_results = results
                
                combined_results.to_csv(csv_path, index=False)
                logger.log(level=logging.INFO, msg=f"Saved evaluation results for {gens_file}.")

        # Save the results for each seed
        c_seed_results.to_csv(os.path.join(c_path, "evaluation_results_all.csv"), index=False)
        logger.log(level=logging.INFO, msg=f"Saved evaluation results for seed {seed}.")
        
    aggregated_results = aggregate_results_across_seeds(all_results)
    
    # Save the aggregated results
    aggregated_results.to_csv(f"{eval_results_dir}/aggregated_evaluation.csv", index=False)

def aggregate_results_across_seeds(combined_results: pd.DataFrame) -> pd.DataFrame:
    grouped = combined_results.groupby(['Model Nickname', 'Aspect'])
    
    # Calculate mean and std for each metric
    summary = grouped['Average'].agg(['mean', 'std']).reset_index()
    summary.rename(columns={'mean': 'Mean Score', 'std': 'Std Score'}, inplace=True)
    # add avrg:
    summary['Mean Score All aspects'] = summary.groupby('Model Nickname')['Mean Score'].transform('mean')
    return summary

def aggregate_results_from_files(path:str)->pd.DataFrame:
    all_results = pd.DataFrame()
    for seed_dir in os.listdir(path):
        if os.path.isdir(f"{path}/{seed_dir}"):
            seed_results = pd.read_csv(f"{path}/{seed_dir}/evaluation_results.csv")
            all_results = pd.concat([all_results, seed_results], ignore_index=True)
    aggregated_results = aggregate_results_across_seeds(all_results)
    return aggregated_results

def evaluate_existing_ds_by_ref(evaluator_name:str,aspect_prompts:dict,ds:Dataset,ref_gens:pd.DataFrame):
    data = _get_dataset_as_prompt_gen_by_ref(ds,ref_gens=ref_gens)
    evaluator_model,tokenizer = _set_evaluator_model(evaluator_name=evaluator_name)
    evaluation_results = evaluate_with_LLM(model=evaluator_model,generated_text= data,aspect_prompts= aspect_prompts,tokenizer= tokenizer)
    # save in csv file:
    evaluation_results.to_csv(f"{REPO_PATH}/generation_output_new_0905/evaluation_results/physics_ds_eval.csv", index=False)

def evaluate_existing_ds(evaluator_name:str,aspect_prompts:dict,ds:Dataset):
    data = _get_dataset_as_prompt_gen(ds)
    evaluator_model,tokenizer = _set_evaluator_model(evaluator_name=evaluator_name)
    evaluation_results = evaluate_with_LLM(model=evaluator_model,generated_text= data,aspect_prompts= aspect_prompts,tokenizer= tokenizer)
    return evaluation_results

if __name__ == "__main__":

    #check if there is args:
    if len(sys.argv) > 1:
        utils.log_w_wandb(logger=logger, message=f"REPO_PATH is provided, evaluating generated data on {sys.argv[1]}")
        gen_folder_path = sys.argv[1]
        agg_resn = main(gen_folder_path)
        # get last nubmer in folder path:
        main(gen_dir_path = gen_folder_path)
    else:
        utils.log_w_wandb(logger=logger, message=f"REPO_PATH is not provided, evaluating domain data")    
        domain_name='Natural_and_physical_sciences__Earth_sciences'
        ds=datasets.load_dataset("machelreid/m2d2", domain_name)
        evaluation_results= evaluate_existing_ds(evaluator_name="mistralai/Mistral-7B-Instruct-v0.2",aspect_prompts=ASPECT_PROMPTS,ds=ds)
        #save in csv file:
        evaluation_results.to_csv(f"{REPO_PATH}/physics_ds_eval.csv", index=False)
