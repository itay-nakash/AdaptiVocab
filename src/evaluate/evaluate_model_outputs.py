import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from domain_generated_tasks.evluate_on_my_tasks import find_response_letter
import evaluate.evaluate_gen_text as evaluate_gen_text
import utils.utils_funcs as utils
from utils.constants import *
import pandas as pd
import argparse
from langchain_google_vertexai import VertexAI
from vertexai import generative_models

def evaluate_generated_data(data,use_gemini:bool=False)->pd.DataFrame:
    gen_text_df=_process_output_to_eval_format(data)
    if use_gemini:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""
        evaluator_model = generative_models.GenerativeModel(model_name="gemini-1.5-pro-preview-0514")
        tokenizer = ''
    else:
        evaluator_model,tokenizer = evaluate_gen_text._set_evaluator_model('mistralai/Mistral-7B-Instruct-v0.2')
    num_examples=None # for debug (trim num of exampels) use 4 here just for debug
    results=evaluate_gen_text.evaluate_with_LLM(model=evaluator_model,tokenizer=tokenizer,generated_text=gen_text_df,aspect_prompts=ASPECT_PROMPTS,num_examples=num_examples)
    results_df = pd.DataFrame({
        'Aspect': results['Aspect'],
        'Average Score': round(results['Average'],4),
        'Standard Deviation': round(results['Std'],4)
    })
    return results_df



def _process_output_to_eval_format(data: Dict) -> pd.DataFrame:
    """Extract prompts and generated texts from the data dictionary and return a DataFrame.
       This version ensures the generated text is truncated to full sentences:
       - If there's a '.' in the generated text, only keep up to and including the last '.'.
       - If no '.' is found, keep the entire text but ensure it ends with '.'.
    """
    prompts = []
    generated_texts = []

    for i in range(len(data['inputs'])):
        prompt = data['inputs'][i]
        text = data['outputs'][i][len(prompt):]

        if '.' in text:
            last_dot_index = text.rfind('.')
            processed_text = text[:last_dot_index+1]
        else:
            # If no '.' found, keep the text as is, but ensure it ends with '.'
            processed_text = text if text.endswith('.') else text + '.'

        prompts.append(prompt)
        generated_texts.append(processed_text)

    df = pd.DataFrame({
        'prompt': prompts,
        'generated_text': generated_texts
    })

    return df


def evaluate_output(file_path,use_gemini:bool=False):
    with open(file_path, 'r') as f:
        data = json.load(f)
        
    if 'generated_data' in file_path:
        return evaluate_generated_data(data,use_gemini)
    elif 'perplexity_data' in file_path:
        return round(data['perplexity'],3)
    else:
        raise ValueError(f'Unexpected file path: {file_path}')

def eval_file(root, file_name, results):
    file_path = os.path.join(root, file_name)
    step = file_name.split('_step_')[-1].split('.')[0]  # Extract the step number
    dataset_name = file_name.split('_output')[0]  # Extract the dataset name
    if '_test_' in file_name:
        dataset_name += '_over_test'
    elif '_validation_' in file_name:
        dataset_name += '_over_validation'
    utils.log_w_wandb(logger=logger, message=f'Evaluating {file_path}', level='info')
    
    if 'generated_data' in file_path and step>540:
        scores = 0
    else:
        # Evaluate the output file
        scores = evaluate_output(file_path)
    
    utils.log_w_wandb(logger=logger, message=f'Finished evaluating {file_path}, scores: {scores}', level='info')
    
    if step not in results:
        results[step] = {}
    results[step][dataset_name] = scores


def evaluate_single_model(output_dir:str,evaluate_generations:bool=False):
    results = {}
    generations_paths=[]
    # Iterate over all JSON files in the output directory
    for root, dirs, files in os.walk(output_dir):
        for file_name in files:
            if file_name.endswith('.json'):
                if 'generated_data' in file_name:
                    generations_paths.append((root,file_name))
                    utils.log_w_wandb(logger=logger,message=f'Added {file_name} to "files to do last"',level='info')
                    continue
                eval_file(root, file_name, results)
                
    utils.log_w_wandb(logger=logger,message=f'Evaluating generations over: {len(generations_paths)} files',level='info')
    if evaluate_generations:
        for root, file_name in generations_paths:
            
            utils.log_w_wandb(logger=logger,message=f'_____________________________________________________________________',level='info')
            utils.log_w_wandb(logger=logger,message=f'Processing {file_name}',level='info')
            utils.log_w_wandb(logger=logger,message=f'_____________________________________________________________________',level='info')
            eval_file(root, file_name, results)
            utils.log_w_wandb(logger=logger,message=f'$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$',level='info')
            utils.log_w_wandb(logger=logger,message=f'Finished processing {file_name}',level='info')
            utils.log_w_wandb(logger=logger,message=f'$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$',level='info')
            utils.log_w_wandb(logger=logger,message=f'reults: {results} ',level='info')
            utils.log_w_wandb(logger=logger,message=f'$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$',level='info')
        utils.log_w_wandb(logger=logger, message=f'Finished Evaluating, results {results}', level='info')
    # Convert the results list to a pandas DataFrame
    df = pd.DataFrame(results).T.reset_index()
    df = df.rename(columns={'index': 'step'})
    df = df.sort_values(by='step', key=lambda x: x.astype(int))

    return df

def evaluate_multiple_outputs_dirs(output_dirs:List[str],csv_output_path:str=None):
    all_results = {}
    max_length=0
    for output_dir in output_dirs:
        model_name = os.path.basename(output_dir.rstrip('/'))
        df = evaluate_single_model(output_dir)
        max_length = max(max_length, len(df))
        for column in df.columns[1:]:  # Skip 'step' column
            all_results[f'{model_name}_{column}'] = df[column].values

    # Padding all arrays to have the same length
    for key in all_results:
        length = len(all_results[key])
        if length < max_length:
            all_results[key] = list(all_results[key]) + [None] * (max_length - length)
    
    aggregated_df = pd.DataFrame(all_results)
    # Add the 'step' column if not already present
    if 'step' not in aggregated_df:
        aggregated_df.insert(0, 'step', range(max_length))
    if csv_output_path:
        os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
        aggregated_df.to_csv(csv_output_path, index=False)

def file_path_to_task_key(file_path:str)->str:
    if 'generated_data' in file_path:
        return 'generated_data'
    
def process_folder_for_eval(folder_paths:List[str],no_gens:bool=False,use_gemini:bool=False):
    for folder_path in folder_paths:
        if not os.path.isdir(folder_path):
            raise ValueError(f"The path {folder_path} is not a folder.")
        
        #Get and add model name:
        folder_parts = folder_path.split('/')
        model_name=''
        for part in folder_parts:
            if part.startswith("model_"):
                model_name = part.split('_', 1)[1]  # Get text after 'model_'
                break
        evaluator='$mistral$' if not use_gemini else '$gemini$'
        output_csv = f"{folder_path}/evaluation_results_{model_name}_{evaluator}_no_gens.csv" if no_gens else f"{folder_path}/evaluation_results_{model_name}_{evaluator}_.csv"
        results = {}
        
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.json'):
                parts = file_name.split('_')
                step_part = parts[-1]
                step = int(step_part.split('.')[0])
                if step>540:
                    continue
                file_path = os.path.join(folder_path, file_name)
                if no_gens and 'generated_data' in file_path:
                    continue
                evaluation_results = evaluate_output(file_path,use_gemini)
                key=file_path_to_task_key(file_path)
                if 'test' in file_path:
                    split='_$test$'
                elif 'validation' in file_path:
                    split='_$validation$'
                else:
                    raise ValueError(f'Unexpected file path: {file_path}')
                if isinstance(evaluation_results, pd.DataFrame):
                    # Loop over aspects in the DataFrame
                    for _, row in evaluation_results.iterrows():
                        aspect = row['Aspect']
                        avg_score = row['Average Score']
                        std_score = row['Standard Deviation']
                        
                        if step not in results:
                            results[step] = {}
                        
                        # Add mean and std for each aspect
                        results[step][f"{aspect}_mean{split}"] = avg_score
                        results[step][f"{aspect}_std{split}"] = std_score
                else:
                    # Handle scalar results
                    if step not in results:
                        results[step] = {}
                    key+=split
                    results[step][key] = evaluation_results
                
                # Log the results
                logger.info(f"Evaluated {file_path} with results: {evaluation_results}")
        
        df = pd.DataFrame.from_dict(results, orient='index').sort_index()
        
        df.to_csv(output_csv)
        print(f"Results saved to {output_csv}")

def evaluate_and_append_step_results(folder_path: str, step: int, no_gens: bool = False):
    if not os.path.isdir(folder_path):
        raise ValueError(f"The path {folder_path} is not a folder.")

    folder_parts = folder_path.split('/')
    model_name = ''
    for part in folder_parts:
        if part.startswith("model_"):
            model_name = part.split('_', 1)[1] 
            break
    output_csv = f"{folder_path}/evaluation_results_{model_name}_no_gens.csv" if no_gens else f"{folder_path}/evaluation_results_{model_name}.csv"
    
    step_results = {}

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json') and f'step_{step}.' in file_name:
            file_path = os.path.join(folder_path, file_name)
            if no_gens and 'generated_data' in file_path:
                continue
            evaluation_results = evaluate_output(file_path)
            key = file_path_to_task_key(file_path)
            
            # Log and store the results for the step
            logger.info(f"Evaluated {file_path} with results: {evaluation_results}")
            step_results[key] = evaluation_results
    
    # Convert step results to DataFrame format
    step_df = pd.DataFrame(step_results, index=[step])
    
    # Load existing results if the CSV file exists, else create a new DataFrame
    if os.path.exists(output_csv):
        existing_df = pd.read_csv(output_csv, index_col=0)
        combined_df = pd.concat([existing_df, step_df], axis=0).sort_index()
    else:
        combined_df = step_df

    # Save the updated results back to the CSV file
    combined_df.to_csv(output_csv)
    print(f"Step {step} results appended to {output_csv}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_paths', type=str, nargs='+', help='Paths to evaluate generations') 
    parser.add_argument('--no_gens', action='store_true', help='Dont eval the gen(takes time)')
    #add arg use gemini defult false if not flag:
    parser.add_argument('--use_gemini', action='store_true', help='Use gemini for evaluation')
    args = parser.parse_args()
    print(args.no_gens)
    process_folder_for_eval(args.eval_paths,args.no_gens,use_gemini=args.use_gemini)