import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.constants import *

from datasets import load_dataset
from transformers import AutoTokenizer
import csv
import utils.utils_funcs as utils
from utils.utils_funcs import log_w_wandb
from collections import Counter
import pandas as pd
import json
import pickle
from collections import defaultdict 
from utils.constants import *


def get_text_from_example(example, text_keys):
    if isinstance(text_keys, list):
        texts = []
        for key in text_keys:
            if key in example:
                value = example[key]
                # Flatten the value if it is a list
                if isinstance(value, list):
                    texts.extend(value)
                else:
                    texts.append(value)
        # Join all text elements into a single string
        text = " ".join(str(text) for text in texts)
    else:
        text = example[text_keys]

    return text

#MACHELREID_M2D2_CONFIGS=['Art', 'Culture_and_the_arts']
dataset_text_keys = {
    "glue": 'sentence',  
    "cnn_dailymail": ['article','highlights'],
    "medmcqa": ['question','opa','opb','opc','opd'],
    "medal": ['text','label'],
} 

class DatasetAnalyzer:
    def __init__(self, tokenizer,max_tokens_to_check:int,n:int,k_most_ngrams:List[int],compared_tokenizer=None):
        self.tokenizer = tokenizer
        self.compared_tokenizer = compared_tokenizer
        self.max_words_to_check = max_tokens_to_check 
        self.n=n
        self.k_most_ngrams=k_most_ngrams

    def analyze_dataset(self, dataset_name:str,text_keys,config_name=None):
        unequal_tokens_list = []
        if config_name == None:
            dataset = load_dataset(dataset_name)
        else:    
            dataset = load_dataset(dataset_name, config_name)
        total_tokens = 0
        total_words = 0
        #ds = dataset['train']
        ds = DatasetAnalyzer._get_ds_val_split(dataset)
        ngrams_list = []

        tokenized_ngrams_dict=defaultdict(lambda: 0)
        tokenized_text = []
        for example in ds:
            if total_words > self.max_words_to_check:
                break
            text = get_text_from_example(example, text_keys)
            tokens = self.tokenizer.tokenize(text)
            if self.compared_tokenizer:
                tokens_compare = self.compared_tokenizer.tokenize(text)
                if tokens != tokens_compare:
                    unequal_tokens_list.append({
                        'text': text,
                        'tokens': tokens,
                        'tokens_compare': tokens_compare
                    })
            total_tokens += len(tokens)
            total_words += len(text.split())
            tokenized_text.append(tokens)
            for t in tokens:
                if type(t)!=str:
                    tokenized_ngrams_dict[t]+=1
            #ngrams_list.extend(list(ngrams(tokens, self.n)))

        if unequal_tokens_list!=[]:
            utils.save_unequal_tokens(unequal_tokens_list)
        #save tokenized_text to file:
        #file_path = ff'{os.path.expanduser("~")}/tokens_count/results/tokenization_test/tokenized_text_{dataset_name}_{config_name}.json'
        #with open(file_path, 'w', encoding='utf-8') as f:
        #    json.dump(tokenized_text, f, ensure_ascii=False, indent=4)
        
        token_to_word_ratio = total_tokens / total_words
        
        # ngrams stats:        
        ngram_counts = Counter(ngrams_list)
        most_common_ngrams={}
        ngrams_ratio={}
        for k in self.k_most_ngrams:
            most_common_ngrams[k] = ngram_counts.most_common(k)
            sum_ngrams_ocur= sum([(count) for ngram, count in most_common_ngrams[k]])
            ngrams_ratio[k] = sum_ngrams_ocur / total_tokens
        
        
        
        return total_tokens, token_to_word_ratio,most_common_ngrams,ngrams_ratio
    
    
    def _get_ds_val_split(dataset):
        if 'validation' in dataset:
            return dataset['validation']
        elif 'validation_matched' in dataset:
            return dataset['validation_matched']
        else:
            raise ValueError("No validation split in dataset")


    
class AnalysisRunner:
    def __init__(self, tokenizer, dataset_names:List[str],n:int=3,k_most_ngrams:List[int]=[],compared_tokenizer=None,max_words_to_check:int = 5900000,ds_configs:List[str]=None):
        self.tokenizer = tokenizer
        if hasattr(tokenizer, 'tokenizer_name'):
            self.run_name = tokenizer.tokenizer_name
        else:
            self.run_name = "pretrain tokenizer"
        self.dataset_analyzer = DatasetAnalyzer(self.tokenizer,max_words_to_check,n,k_most_ngrams,compared_tokenizer)
        self.dataset_names = dataset_names
        self.results = {}
        self.ds_configs=ds_configs

    def run_analysis(self,save_every:int=20):
        for dataset_name in self.dataset_names:
            text_key = dataset_text_keys.get(dataset_name, 'text')

            if dataset_name == "glue":
                config_names = ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]
                text_keys = ['sentence',['premise','hypothesis'],['sentence1','sentence2'],['question','sentence'],['question1','question2'],['sentence1','sentence2'],'sentence',['sentence1','sentence2'],['sentence1','sentence2']]
                for config_name,text_key in zip(config_names,text_keys):
                    self._run_and_log_results(dataset_name,text_key,config_name,self.run_name)
            elif dataset_name == "cnn_dailymail":
                # Specify the config name for cnn_dailymail. Example: '3.0.0'
                config_name = '3.0.0'
                self._run_and_log_results(dataset_name,text_key,config_name,self.run_name)
            elif dataset_name == "machelreid/m2d2":
                num_of_runs=0
                self.ds_configs = MACHELREID_M2D2_CONFIGS if self.ds_configs is None else self.ds_configs
                for config_name in self.ds_configs:
                    try:
                        self._run_and_log_results(dataset_name,text_key,config_name,self.run_name)
                        #self.count_num_of_words(dataset_name,text_key,config_name)
                        num_of_runs+=1
                        if num_of_runs%save_every==0:
                            self.export_results_to_csv(f'{os.path.expanduser("~")}/tokens_count/results/ds_stats/ds_stats{num_of_runs}_count_words.csv')
                    except Exception as e:
                        print(f"Failed on config {config_name} with error: {e}")
                        # Optionally, log the error or handle it in some way here
                        continue
            else:
                # For datasets without specific config names
                config_name = ''
                self._run_and_log_results(dataset_name,text_key,config_name,self.run_name)
        return self.results

    def export_results_to_csv(self,path:str):
        with open(path, 'w', newline='') as file:
            writer = csv.writer(file)
            data_keys = list(self.results[next(iter(self.results))].keys())
            writer.writerow(data_keys)
            for dataset_name, data in self.results.items():
                data_values = [dataset_name] + [v for k,v in data.items()]
                writer.writerow(data_values)
                #most_common_ngrams_str = ', '.join([f"{ngram}: {count}" for ngram, count in data.get('Most Common Ngrams')])
                #ngram_ratios_str = ', '.join([f"{scngram}: {ratio:.3f}" for ngram, ratio in data.get('Ngrams Ratios')])

    def count_num_of_words(self,dataset_name,text_key,config_name):
        dataset = load_dataset(dataset_name, config_name)
        total_words = 0
        results_key = f"{dataset_name}_{config_name}"
        self.results[results_key]={}
        for example in dataset['train']:
            text = get_text_from_example(example, text_key)
            total_words += len(text.split())
        self.results[results_key]['number_of_words'] = total_words
    
    def _run_and_log_results(self,dataset_name,text_key,config_name,tokenizer_name):
        total_tokens, ratio, most_common_ngrams,ngrams_ratios = self.dataset_analyzer.analyze_dataset(dataset_name,text_key,config_name)
        results_key = f"{dataset_name}_{config_name}"
        self.results[results_key] = {'Total Tokens': total_tokens, 'Token to Word Ratio': ratio}
        for k,k_ratio in ngrams_ratios.items():
            self.results[results_key][f'{k} Most common ratio'] = k_ratio
            number_of_ngrams=k_ratio*results_key['Total Tokens']
            ngrams_savings=number_of_ngrams*2 # FOR 3-GRAMS ONLY
            self.results[results_key][f'{k} Most common saving'] = ngrams_savings
        log_w_wandb(logger=logger,
                    message=("Dataset: " + dataset_name + ", "
                            "Total Tokens: " + f"{total_tokens:,}" + " Token to Word Ratio: " + f"{ratio:.3f}" + " "
                            ),
                    kwargs={f'{tokenizer_name}/{dataset_name}/total_tokens': total_tokens,
                            f'{tokenizer_name}/{dataset_name}/ratio': ratio,
                            f'{tokenizer_name}/{dataset_name}/most_common_ngrams': most_common_ngrams,
                            f'{tokenizer_name}/{dataset_name}/ngrams_ratios': ngrams_ratios,
                            })


def compare_results(results_before,results_after):
    for key in results_before.keys():
        log_data = {
        'dataset': key,
        'total_tokens': {
            'before': results_before[key]['Total Tokens'],
            'after': results_after[key]['Total Tokens']
        },
        'token_to_word_ratio': {
            'before': results_before[key]['Token to Word Ratio'],
            'after': results_after[key]['Token to Word Ratio'],
            'diff': results_after[key]['Token to Word Ratio'] - results_before[key]['Token to Word Ratio'],
            'diff_normalized': (results_after[key]['Token to Word Ratio'] - results_before[key]['Token to Word Ratio']) / results_before[key]['Token to Word Ratio']
        }
        }
    

        msg=f"Dataset: {key}, Total Tokens before: {results_before[key]['Total Tokens']:,}\
            Token to Word Ratio before: {results_before[key]['Token to Word Ratio']:.3f} Total Tokens after: {results_after[key]['Total Tokens']:,}\
                Token to Word Ratio after: {results_after[key]['Token to Word Ratio']:.3f}, Token to word ratio diff: {results_after[key]['Token to Word Ratio'] - results_before[key]['Token to Word Ratio']:.3f}\
                    ,'Ratio_diff_normalized':{round((results_after[key]['Token to Word Ratio'] - results_before[key]['Token to Word Ratio'])/results_before[key]['Token to Word Ratio'])}"
        log_w_wandb(logger=logger,message=msg,kwargs=log_data)

def save_results(reduced_tokens,added_ngrams,results_before_reduce,results_after_reduce,results_after_adding_ngrams,cfg):
    #create folder for run:
    folder_path= os.path.join(utils.adjust_path_to_server(f'{os.path.expanduser("~")}/tokens_count/results/spesific_run_results2'),utils.cfg_to_name_add_ngrams(cfg=cfg,file_type=None,num_of_ngrams=100*cfg['num_to_add']))
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    #save reduced tokens:
    file_path = os.path.join(folder_path,'reduced_tokens.json')
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(list(reduced_tokens), f, ensure_ascii=False, indent=4)
    
    #save added ngrams:
    file_path = os.path.join(folder_path,'added_ngrams.pkl')
    with open(file_path, 'wb') as file:
        pickle.dump(added_ngrams, file)
        
    file_name = utils.cfg_to_name_add_ngrams(cfg=cfg,file_type=None,num_of_ngrams=100*cfg['num_to_add'])
    #convert to df:
    df_before_reduce = pd.DataFrame.from_dict(results_before_reduce,orient='index')
    df_after_reduce = pd.DataFrame.from_dict(results_after_reduce,orient='index')
    df_after_adding_ngrams = pd.DataFrame.from_dict(results_after_adding_ngrams,orient='index')
    
    #save to csv:
    df_before_reduce.to_csv(os.path.join(folder_path,f'df_{file_name}_before_reduce.csv'))
    df_after_reduce.to_csv(os.path.join(folder_path,f'df_{file_name}_after_reduce.csv'))
    df_after_adding_ngrams.to_csv(os.path.join(folder_path,f'df_{file_name}_after_adding_ngrams.csv'))


if __name__ == "__main__":
    model_name = "facebook/opt-350m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset_names = ['machelreid/m2d2']
    k_most_ngrams=[100,500,1000,2000,3000,4000,5000,10000,20000,30000,50000,100000,200000]
    runner = AnalysisRunner(tokenizer, dataset_names,k_most_ngrams=k_most_ngrams,max_words_to_check=900000)
    runner.run_analysis()

