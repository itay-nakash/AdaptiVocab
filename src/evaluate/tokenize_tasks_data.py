import json
from tqdm import tqdm
import csv
import os
import sys
from transformers import AutoTokenizer
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.constants import *
from datasets import Dataset
import utils.utils_funcs as utils
from build_vocab.patch_tokenizer import PatchTokenizer
import argparse
from src.ft.dp_train import DataPrepare


pattern = (
	"You will receive multiple-choice questions presented as follows: 'Instruction: '. "
	"Your task is to respond with a detailed explanation and then provide the correct answer. "
	"Format your answer like this, substituting <YOUR_ANSWER> with the selected option from the provided choices: "
	"'The answer is: <YOUR_ANSWER>.'\n\n"
	"### Instruction:\n{}\n\n### Response: Let's analyze this step by step.\n"
)

def prepare_data_for_generation(tok_dataset, tokenized_path: str, tokenizer, examples_to_generate: int=500):
	# Prepare data and counter
	inputs_list=[]
	num_of_examples = 0
	# Iterate over the dataset
	for example in tok_dataset:
		if num_of_examples >= examples_to_generate:
			break

		if len(example['input_ids']) < MIN_LEN_INPUT:
			continue
		full_decoded = tokenizer.decode(example['input_ids'])
		first_dot_idx = full_decoded.find('.')
		if first_dot_idx != -1:
			processed_text = full_decoded[first_dot_idx+1:].strip()
		else:
			processed_text = full_decoded.strip()
		
		# Re-encode after removing the prefix before the first dot
		full_input = tokenizer(processed_text, padding=False, truncation=False)['input_ids']

		# Determine the sequence length based on the preset limits or available tokens
		print(f'len prompt_len_list: {len(PROMPT_LEN_LIST)}, exmple num: {num_of_examples}')
		seq_len = min(PROMPT_LEN_LIST[num_of_examples], len(full_input))
		cur_input = full_input[:seq_len]
		# Append the current input and its length to the prepared data list

		current_input={
      'prompt':tokenizer.decode(cur_input[:seq_len]),
      'input_ids':cur_input,
      'correct_answer':'None'
      }
		inputs_list.append(current_input)
		num_of_examples += 1
		#log the generated prompt:
		utils.log_w_wandb(logger=logger, message=f"prompt: {current_input['prompt']}")
		utils.log_w_wandb(logger=logger, message=f"input_ids: {current_input['input_ids']}")
	# Create the directory if it doesn't exist
	os.makedirs(os.path.dirname(tokenized_path), exist_ok=True)
#
	## Save the prepared data to a JSON file
	with open(tokenized_path, 'w', encoding='utf-8') as file:
		json.dump(inputs_list, file, ensure_ascii=False, indent=4)
#
	## Optionally log the completion of the preparation
	utils.log_w_wandb(logger=logger, message=f"Preparation for data generation completed, examples prepared: {num_of_examples}")

def get_task_name(tokenizer_p:str):
	if 'Natural_and_physical_sciences__Earth_sciences' in tokenizer_p:
		task_name = "Natural_and_physical_sciences__Earth_sciences"
	elif 'Culture_and_the_arts__Games_and_Toys' in tokenizer_p:
		task_name = "Culture_and_the_arts__Games_and_Toys"
	elif 'physics.hist-ph' in tokenizer_p:
		task_name = "physics.hist-ph"
	elif 'Culture_and_the_arts__The_arts_and_Entertainment' in tokenizer_p:
		task_name = "Culture_and_the_arts__The_arts_and_Entertainment"
	elif 'Health_and_fitness' in tokenizer_p:
		task_name = "Health_and_fitness"
	return	task_name


def prepare_data_for_generation_only(base_tokenizer: str, tokenizers_paths, examples_to_generate: int = 500):
    """Prepare data for generation from each tokenizer's tokenized dataset.
    
    Args:
        base_tokenizer (str): The path or name of the base tokenizer.
        tokenizers_paths (List[str]): A list of paths to different tokenizers.
        examples_to_generate (int): How many examples to generate. Defaults to 500.
    """
    # Log tokenizers paths (assuming utils.log_w_wandb and logger are defined elsewhere)
    utils.log_w_wandb(logger=logger, message=f"Tokenizers paths: {tokenizers_paths}")
    
    for tokenizer_p in tokenizers_paths:

        # Load tokenizer
        if '_vanila' in tokenizer_p:
            tokenizer = AutoTokenizer.from_pretrained(base_tokenizer)
        else:
            tokenizer = PatchTokenizer.load_model_from_scratch(
                path=os.path.join(tokenizer_p,'patch_tokenizer.pkl'),
                existing_tokenizer_name=base_tokenizer
            )

        tokenized_tasks_path = f"{tokenizer_p}/tokenized_tasks/"
        assert os.path.exists(f"{tokenizer_p}/tokenized_datasets/"), f"Tokenized datasets not found for {tokenizer_p}"
        # Load tokenized dataset (assuming utils.load_pickle is defined)
        tok_data_path = f"{tokenizer_p}/tokenized_datasets/machelreid_m2d2_tokenized_lm_dataset_val10000.pkl"
        tok_examples = utils.load_pickle(tok_data_path)

        # Prepare data for generation only
        # Assuming prepare_data_for_generation is defined elsewhere
        prepare_data_for_generation(tok_examples, f"{tokenized_tasks_path}/generated_data.json",tokenizer, examples_to_generate=500)


def prepare_all_data(base_tokenizer:str,tokenizers_paths):
	#tokenizers_paths = [(f'{base_tokenizer_path}/{f}',f'{base_tokenizer_path}/{f}/tokenized_datasets') for f in os.listdir(base_tokenizer_path) if os.path.isdir(os.path.join(base_tokenizer_path, f))]
	utils.log_w_wandb(logger=logger, message=f"Tokenizers paths: {tokenizers_paths}")
	
	#create folder if not exists:
	for tokenizer_p in tokenizers_paths:
		task_name=get_task_name(tokenizer_p)
		if '_vanila' in tokenizer_p:
			tokenizer = AutoTokenizer.from_pretrained(base_tokenizer)
   
		else:
			tokenizer = PatchTokenizer.load_model_from_scratch(path = os.path.join(tokenizer_p,'patch_tokenizer.pkl') ,
																	existing_tokenizer_name=base_tokenizer)

	# Need to make this work
 #	 if not os.path.exists(f"{tokenizer_p}/tokenized_datasets/"):
	#		logger.info(f"Tokenizing data for {tokenizer_p}")
	#		os.makedirs(f"{tokenizer_p}/tokenized_datasets/", exist_ok=False)
	#		prepare_tokenized_data(tokenizer_p, f'machelreid_m2d2/{task_name}',tokenizer)

		tokenized_tasks_path = f"{tokenizer_p}/tokenized_tasks/"
		os.makedirs(tokenized_tasks_path, exist_ok=False)  
	#	prepare_data_for_qa(tokenized_tasks_path+'/qa/',tokenizer)
		tok_data_path=f"{tokenizer_p}/tokenized_datasets/machelreid_m2d2_tokenized_lm_dataset_val10000.pkl"
		tok_examples = utils.load_pickle(tok_data_path)
		prepare_data_for_generation(tok_examples, f"{tokenized_tasks_path}/generated_data.json",tokenizer, examples_to_generate=500)
		
  		# currently not doing preplexity:
		prepare_preplexity_data(tok_examples, f"{tokenized_tasks_path}/perplexity_data.json", examples_to_generate=500, wanted_len=256)		
  
  		# load my task data as json from /data/home/itay.nakash/patch_tokenizer/domain_generated_tasks/processed_domain_tasks_randomized_order.json
		tasks_qa_context_path=f"{REPO_PATH}/src/domain_generated_tasks/qustions_path/{task_name}/randomized/domain_qa_context_over_test_randomized_order.json"
		tasks_qa_long_path=f"{REPO_PATH}/src/domain_generated_tasks/qustions_path/{task_name}/randomized/domain_qa_no_context_over_test_randomized_order.json"
		tasks_qa_context_over_train=f"{REPO_PATH}/src/domain_generated_tasks/qustions_path/{task_name}/randomized/domain_qa_context_over_train_randomized_order.json"
		tasks_qa_no_context_over_train=f"{REPO_PATH}/src/domain_generated_tasks/qustions_path/{task_name}/randomized/domain_qa_context_over_test_randomized_order.json"
		with open(tasks_qa_context_path, 'r') as file:
			data_context = json.load(file)
		with open(tasks_qa_long_path, 'r') as file:
			data_long = json.load(file)
		with open(tasks_qa_context_over_train, 'r') as file:
			data_context_over_train = json.load(file)
		with open(tasks_qa_no_context_over_train, 'r') as file:
			data_no_context_over_train = json.load(file)


  
		prepare_prompts_my_task(data_context, tokenizer, PROMPT_MY_TASKS_PHYSICS, f"{tokenized_tasks_path}/domain_qa_context_tokenized.json", use_context=True)
		prepare_prompts_my_task(data_context, tokenizer, PROMPT_MY_TASKS_PHYSICS_NO_CONTEXT, f"{tokenized_tasks_path}/domain_qa_no_context_tokenized.json", use_context=False)
		prepare_prompts_my_task(data_long, tokenizer, PROMPT_MY_TASKS_PHYSICS_NO_CONTEXT, f"{tokenized_tasks_path}/domain_qa_no_context_long_tokenized.json", use_context=False)
		prepare_prompts_my_task(data_context_over_train, tokenizer, PROMPT_MY_TASKS_PHYSICS, f"{tokenized_tasks_path}/domain_qa_context_over_train_tokenized.json", use_context=True)
		prepare_prompts_my_task(data_no_context_over_train, tokenizer, PROMPT_MY_TASKS_PHYSICS_NO_CONTEXT, f"{tokenized_tasks_path}/domain_qa_no_context_over_train_tokenized.json", use_context=False)
		split_train_val(tokenized_tasks_path)


def prepare_tokenized_data(tokenizer_path:str,target_corpus_name:str,tokenizer):
    raise NotImplementedError("This function is not implemented yet")
    path_to_tokenize_ds=os.path.join(tokenizer_path,'tokenized_datasets')
    ds_names, ds_configs = utils.get_ds_config_names(target_corpus_name)
    ds_name=ds_names[0] if type(ds_names)==list else ds_names
    ds_config=ds_configs[0] if type(ds_configs)==list else ds_configs
    logger.info(f"loading and preparing data for {ds_name}")
    dp=DataPrepare(dataset_name=ds_name, tokenizer=tokenizer,dataset_config=ds_config)
    lm_dataset=dp.load_and_prepare_data(sys.maxsize, 10_000,tokenized_data_path=path_to_tokenize_ds)
    return lm_dataset

  
def split_train_val(path:str):
    val_ratio = 0.2
    
    #make validation and train paths:
    os.makedirs(f"{path}/test", exist_ok=True)
    os.makedirs(f"{path}/validation", exist_ok=True)
    # for file in folder:
    for file in os.listdir(path):
        #if not directory:
        if os.path.isdir(f'{path}/{file}'):
           continue
        with open(f"{path}/{file}", 'r') as f:
            data = json.load(f)
            # Split the data into train and validation sets
            val_size = int(val_ratio * len(data))
            train_data = data[val_size:]
            val_data = data[:val_size]
            # Write the train and validation data to separate files
            with open(f"{path}/test/{file}", 'w') as f:
                json.dump(train_data, f, indent=4)	
            with open(f"{path}/validation/{file}", 'w') as f:
                json.dump(val_data, f, indent=4)    


def main():
    utils.set_seed(SEED)
    debug = True

    if debug:
        tokenizer_path = '/data/home/itay.nakash/patch_tokenizer/src/saved_patch_tokenizers/original_tokenizer_meta-llama_Llama-2-7b-hf_target_corpus_name_machelreid_m2d2_Culture_and_the_arts__Games_and_Toys_num_to_add_10000_num_to_remove_10000_ngram_k_3_/'
        tokenizers_paths = tokenizer_path
    else:
        parser = argparse.ArgumentParser(description='Prepare data for tokenizers.')
        parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to the tokenizer directory')
        args = parser.parse_args()
        tokenizer_path = args.tokenizer_path
    
    tokenizers_paths = [tokenizer_path]
    tokenized_tasks_path = f"{tokenizer_path}/tokenized_tasks/"

    if 'Mistral-7B-v0.1' in tokenizer_path:
        base_tokenizer = 'mistralai/Mistral-7B-v0.1'
    elif 'Mistral-7B-v0.3' in tokenizer_path:
        base_tokenizer = 'mistralai/Mistral-7B-v0.3'
    elif 'meta-llama_Llama-2-7b':
        base_tokenizer = 'meta-llama/Llama-2-7b-hf'
    else:
        raise ValueError(f"Tokenizer {tokenizer_path} is not supported")

    prepare_all_data(base_tokenizer, tokenizers_paths)
    split_train_val(tokenized_tasks_path)


if __name__ == '__main__':
    main()





