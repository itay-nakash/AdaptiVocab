import pickle
from datasets import DatasetDict,Dataset
import pickle
import os
import utils.utils_funcs as utils
from datasets import load_dataset, concatenate_datasets
import json
from tqdm import tqdm
from utils.constants import *


class DataPrepare:
    def __init__(self, dataset_name: str, tokenizer,dataset_config:str =None  ,num_samples: int = None):
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.tokenized_lm_dataset = {}
    
    def load_and_prepare_data(self, num_train_samples:int=100, num_val_samples:int=20,tokenized_data_path:str=None,example_len:int=768):
        # check if the tokenized dataset is already exists:
        safe_ds_name = self.dataset_name.replace('/','_')
        self.ds = load_dataset(self.dataset_name,self.dataset_config)
        num_train_samples = min(num_train_samples, len(self.ds['train']))
        num_val_samples = min(num_val_samples, len(self.ds['validation']),MAX_VALIDATION_SIZE)
        train_ds_path=os.path.join(tokenized_data_path,f'{safe_ds_name}_tokenized_lm_dataset_train{num_train_samples}.pkl')
        val_ds_path=os.path.join(tokenized_data_path,f'{safe_ds_name}_tokenized_lm_dataset_val{num_val_samples}.pkl')
       
        if tokenized_data_path and os.path.exists(train_ds_path):
            utils.log_w_wandb(logger=logger,message=f"Loading tokenized train dataset from {train_ds_path}")
            if not os.path.exists(val_ds_path):
                utils.log_w_wandb(logger=logger,message=f"Tokenizing val to {val_ds_path}")
                self.ds['validation'] = self.ds['validation'].select(range(num_val_samples))
                ds_validation=Dataset.from_dict(self.merge_and_tokenize(self.ds['validation'], example_len=example_len))
                pickle.dump(ds_validation, open(val_ds_path, 'wb'))
            return self.load_tokenized_data(train_ds_path,val_ds_path)
        
        utils.log_w_wandb(logger=logger,message=f"Tokenizing dataset {self.dataset_name}")
        self.ds['train'] = self.ds['train'].select(range(num_train_samples))
        self.ds['validation'] = self.ds['validation'].select(range(num_val_samples))
        if 'full' in self.ds:
            del self.ds['full']
        if 'test' in self.ds:
            del self.ds['test']
            
        logger.info(f"Loaded dataset {self.dataset_name} config: {self.dataset_config} with train size {len(self.ds['train'])} and validation size {len(self.ds['validation'])}")
        logger.info(f"number of words in train: {sum(len(text.split()) for text in self.ds['train']['text'])}")
        logger.info(f'prepare train dataset:')
        self.tokenized_lm_dataset['train'] = Dataset.from_dict(self.merge_and_tokenize(self.ds['train'], example_len=example_len))
        logger.info(f'finished preparing train dataset')
        utils.log_w_wandb(logger=logger,message=f"train dataset size is {len(self.tokenized_lm_dataset['train'])}")
        logger.info(f'prepare validation dataset:')
        self.tokenized_lm_dataset['validation'] = Dataset.from_dict(self.merge_and_tokenize(self.ds['validation'], example_len=example_len))
        logger.info(f'finished preparing validation dataset')
        
        # save the tokenized dataset:
        if tokenized_data_path:
            # Ensure the directory exists
            os.makedirs(tokenized_data_path, exist_ok=True)
            
            # Save the datasets
            train_path = os.path.join(tokenized_data_path, f'{safe_ds_name}_tokenized_lm_dataset_train{num_train_samples}.pkl')
            validation_path = os.path.join(tokenized_data_path, f'{safe_ds_name}_tokenized_lm_dataset_val{num_val_samples}.pkl')
            pickle.dump(self.tokenized_lm_dataset['train'], open(train_path, 'wb'))
            pickle.dump(self.tokenized_lm_dataset['validation'], open(validation_path, 'wb'))
        
        self.tokenized_lm_dataset = DatasetDict({split: dataset for split, dataset in self.tokenized_lm_dataset.items()})
        self.tokenized_lm_dataset = utils.correct_nesting_for_dataset(self.tokenized_lm_dataset)

        return self.tokenized_lm_dataset
    
    
    def merge_and_tokenize(self,examples: Dict[str, List[str]], example_len: int = 256) -> Dict[str, List[int]]:
        assert self.tokenizer is not None, "The tokenizer must be set - the tokenized data is not exists"
        assert self.tokenizer.bos_token_id is not None, "The tokenizer must have a bos_token_id"
        assert self.tokenizer.eos_token_id is not None, "The tokenizer must have an eos_token_id"
        tokenized_texts = []
        utils.log_w_wandb(logger=logger,message=f"Current Dataset Len: {len(examples['text'])}")
        texts = [text for text in examples["text"] if len(text.split()) > MIN_EXAMPLE_LEN]
        utils.log_w_wandb(logger=logger,message=f"Filtered above len {MIN_EXAMPLE_LEN} Dataset Len: {len(texts)}")
        for text in tqdm(texts, desc="Tokenizing"):
            tokenized_texts.append(self.tokenizer(text, padding=False, truncation=False))
            orignial_texts = [text for text in texts] #$$$$ maybe add $$$$
        # Initialize storage for chunks
        batched_exmps = {'input_ids': [], 'attention_mask': [], 'labels': [],'original_text':[]}
        current_chunk = []
        for row in tqdm(tokenized_texts,desc='Tokenizing and chuncking'):
            for token_id in row['input_ids']:
                current_chunk.append(token_id)
                if len(current_chunk) == example_len:
                    batched_exmps['input_ids'].append(current_chunk)
                    batched_exmps['attention_mask'].append([1] * example_len)
                    batched_exmps['labels'].append(current_chunk.copy())
                    batched_exmps['original_text'].append(orignial_texts.pop(0))# $$$$ maybe add $$$$
                    current_chunk = []
                    
        return batched_exmps

    def filter_empty_examples(dataset):
        # Assuming dataset is a Hugging Face 'datasets.Dataset' instance
        return dataset.filter(lambda example: len(example['input_ids']) > 0)

    def load_tokenized_data(self,train_ds_path:str, val_ds_path:str):
        utils.log_w_wandb(logger=logger,message=f"Loading tokenized dataset from {train_ds_path} and {val_ds_path}")
        # Initialize an empty DatasetDict
        self.tokenized_lm_dataset = DatasetDict()

        train_dataset = pickle.load(open(train_ds_path, 'rb'))
        self.tokenized_lm_dataset['train'] = train_dataset

        val_dataset = pickle.load(open(val_ds_path, 'rb'))
        self.tokenized_lm_dataset['validation'] = val_dataset

        return self.tokenized_lm_dataset

    def load_and_merge_json(self, existing_ds: DatasetDict,tokenized_data_path:str):
        # Load and tokenize JSON data
        ds_qa = []
        safe_ds_name = self.dataset_name.replace('/','_')
        #self.ds = load_dataset(self.dataset_name,self.dataset_config)
        train_ds_path=os.path.join(tokenized_data_path,f'{safe_ds_name}_tokenized_QA_dataset.pkl')
        if os.path.exists(train_ds_path):
            utils.log_w_wandb(logger=logger,message=f"Loading tokenized train dataset from {train_ds_path}")
            ds_qa = pickle.load(open(train_ds_path, 'rb'))
        else:
            for path in TASKS_DATA_PATHS:
                with open(path, 'r') as file:
                    json_data = [json.loads(line) for line in file]

                if utils.is_debug_mode():
                    json_data = json_data[:10]
                # Tokenize the data
                tokenized_data = [self.tokenizer(data['problem'] + " " + data['solution']) for data in json_data]
                tokenized_dataset = Dataset.from_dict({'input_ids': [d['input_ids'] for d in tokenized_data], 'attention_mask': [d['attention_mask'] for d in tokenized_data], 'labels': [d['input_ids'] for d in tokenized_data]})
                ds_qa.append(tokenized_dataset)
            # save_to_pkl:
            os.makedirs(tokenized_data_path, exist_ok=True)
            pickle.dump(ds_qa, open(train_ds_path, 'wb'))

        # Merge the JSON datasets with the existing dataset
        combined_data=existing_ds['train']
        for new_ds in ds_qa:
            combined_data = concatenate_datasets([combined_data, new_ds])
        combined_data=combined_data.shuffle(seed=SEED)
        return combined_data

def get_tokenized_ds(cfg:Dict,tokenizer='None') -> DatasetDict:
    path_to_tokenize_ds=os.path.join(cfg['tokenizer_path'],'tokenized_datasets') if cfg['tokenizer_path'] else None
    ds_names, ds_configs = utils.get_ds_config_names(cfg['target_corpus_name'])
    ds_name=ds_names[0] if type(ds_names)==list else ds_names
    ds_config=ds_configs[0] if type(ds_configs)==list else ds_configs
    dp=DataPrepare(dataset_name=ds_name, tokenizer=tokenizer,dataset_config=ds_config)
    lm_dataset=dp.load_and_prepare_data(cfg['example_to_train'], cfg['example_to_val'],tokenized_data_path=path_to_tokenize_ds)
    return lm_dataset

#
#def get_tasks_tokenized_ds(cfg:Dict,tokenizer='None') -> DatasetDict:
#    path_to_qustions = f'{REPO_PATH}/domain_generated_tasks/qustions_path/{cfg['target_corpus_name']}/randomized/domain_qa_context_over_train_randomized_order.json'
#    path_to_tldr = f'{REPO_PATH}/domain_generated_tasks/tldr_path'
#    path_to_tokenize_ds=os.path.join(cfg['tokenizer_path'],'tokenized_datasets','tasks') if cfg['tokenizer_path'] else None
#    ds_names, ds_configs = utils.get_ds_config_names(cfg['target_corpus_name'])
#    ds_name=ds_names[0] if type(ds_names)==list else ds_names
#    ds_config=ds_configs[0] if type(ds_configs)==list else ds_configs
#    dp=DataPrepare(dataset_name=ds_name, tokenizer=tokenizer,dataset_config=ds_config)
#    lm_dataset=dp.load_and_merge_json(cfg['example_to_train'], cfg['example_to_val'],tokenized_data_path=path_to_tokenize_ds)
#    return lm_dataset
#

def get_kd_tokenized_ds(cfg:Dict,tokenizer='None') -> DatasetDict:
    path_to_tokenize_ds=os.path.join(cfg['tokenizer_path'],'tokenized_datasets','kd') if cfg['tokenizer_path'] else None
    ds_names, ds_configs = utils.get_ds_config_names(cfg['target_corpus_name'])
    ds_name=ds_names[0] if type(ds_names)==list else ds_names
    ds_config=ds_configs[0] if type(ds_configs)==list else ds_configs
    dp=DataPrepare(dataset_name=ds_name, tokenizer=tokenizer,dataset_config=ds_config)
    lm_dataset=dp.load_and_prepare_data(cfg['example_to_train'], cfg['example_to_val'],tokenized_data_path=path_to_tokenize_ds)
    return lm_dataset