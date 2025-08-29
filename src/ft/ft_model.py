import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import wandb
from transformers import AutoTokenizer,AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import DatasetDict, Dataset
from utils.constants import *
import dp_train
from utils import utils_funcs as utils
from build_vocab.load_custom_model import load_custom_model, load_custom_model_kd, Evaluator
from kd_code import custom_trainer_kd
import evaluate.QA.eval
from build_vocab.patch_tokenizer import PatchTokenizer
from transformers import TrainerCallback
import evaluate.evaluate_on_tokenized_data
from transformers import TrainerCallback, TrainerState, TrainerControl
import json
import torch
from transformers import AutoTokenizer
from typing import Dict
from kd_code import custom_trainer_kd
from evaluate import evaluate_model_outputs

class EvaluationCallback(TrainerCallback):
    def __init__(self, eval_datasets, output_paths,tokenizer,do_val=True,do_test=True, eval_every_k_steps=100):
        self.eval_datasets = eval_datasets
        self.output_paths = output_paths
        self.eval_every_k_steps = eval_every_k_steps
        self.tokenizer = tokenizer
        self.do_val=do_val 
        self.do_test=do_test

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        heval=False
        if state.global_step % self.eval_every_k_steps == 0:
            val_output_paths = {name: f"{self.output_paths['validation'][name]}_validation" for name in self.output_paths['validation']}
            test_output_paths = {name: f"{self.output_paths['test'][name]}_test" for name in self.output_paths['test']}
            if heval:
                    evaluate.evaluate_on_tokenized_data.evaluate_and_save_he(path='/data/home/itay.nakash/patch_tokenizer/src/h_eval_data/hist_science/tokenized_data.json',tokenizer=self.tokenizer,global_step=state.global_step,
                                                                             model=kwargs['model'],output_path='/data/home/itay.nakash/patch_tokenizer/src/h_eval_data/hist_science/gens.json')
            else:
                
                if self.do_val:         
                    evaluate.evaluate_on_tokenized_data.evaluate_and_save(eval_datasets=self.eval_datasets['validation'],tokenizer=self.tokenizer,
                                                                        global_step=state.global_step,model=kwargs['model'],
                                                                        output_paths=val_output_paths)
                if self.do_test:
                    evaluate.evaluate_on_tokenized_data.evaluate_and_save(eval_datasets=self.eval_datasets['test'],tokenizer=self.tokenizer,
                                                                        global_step=state.global_step,model=kwargs['model'],
                                                                        output_paths=test_output_paths)
                logger.info(f"Finished generating answers and generation to step {state.global_step}.")

class NLPModelTrainer:
    def __init__(self, cfg:Dict,tokenizer=None,model=None):
        self.model_name = cfg['original_tokenizer']
        if model != None:
            self.model = model
        else:
            if cfg['kd'] and False:
                model_to_load='/data/home/itay.nakash/patch_tokenizer/good_trained_models_0506/model_7cfcc001c1_sweep_usual-puddle-578/checkpoint-700'
                self.model = load_custom_model_kd(model_to_load=model_to_load,emb_method=cfg['emb_method'],patch_tokenizer_path=cfg['tokenizer_path'],unfroz=cfg['unfrozen'])
            else:
                self.model = load_custom_model(model_name=cfg['original_tokenizer'],emb_method=cfg['emb_method'],patch_tokenizer_path=cfg['tokenizer_path'],use_lora=cfg['use_lora'],unfroz=cfg['unfrozen'])
        self.trainer = None
        self.tokenizer = tokenizer
    
    def train(self, cfg:Dict,tokenized_lm_dataset:Dataset):
        do_eval_zero_step=False
        output_dir=utils.cfg_to_filename(cfg)
        hased_output_dir=f"./trained_models_0506/{utils.hash_output_dir(output_dir)}_sweep_{wandb.run.name}"
        
        save_every_steps = 20
        utils.log_w_wandb(logger=logger, message=f"save every steps: {save_every_steps}")
        training_args = TrainingArguments(
            output_dir=hased_output_dir,
            eval_strategy="epoch",
            learning_rate=cfg['lr'],
            weight_decay=cfg['weight_decay'],
            warmup_ratio=cfg['warmup_ratio'],
            num_train_epochs=cfg['num_epochs'],
            save_strategy="no",  # Changed to 'steps'
            save_steps=save_every_steps,  # Set to save every half epoch
            per_device_train_batch_size = cfg['batch_size'],
            per_device_eval_batch_size = cfg['batch_size'],
            gradient_accumulation_steps=cfg['gradient_accumulation_steps'],
            report_to="wandb",
            push_to_hub=False,
        )
        
        
        tokenized_lm_dataset['train'].set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels']) 
        tokenized_lm_dataset['validation'].set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        if cfg['torch_dtype']==torch.bfloat16:
            self.model = self.model.bfloat16()
        # assert sizes and shapes, len each example is correct
        #tokenizers_tests.assert_input_sizes(tokenized_lm_dataset,tokenized_lm_dataset['train']['input_ids'].shape[1])
       
        logger.info(f"Training model {self.model_name} on dataset {cfg['original_tokenizer']} of train size {len(tokenized_lm_dataset['train'])}...")
        
        eval_datasets = utils.get_eval_ds_path(cfg['tokenizer_path'])
        model_output_dir = f"{EVAL_OUTPUTS_PATH}/layers_ablation/first_two/physical/{hased_output_dir}_outputs"
        os.makedirs(model_output_dir, exist_ok=True)
        output_paths={}
        output_paths['validation'] = {name: f"{model_output_dir}/{name}_output.json" for name in eval_datasets['validation']}
        output_paths['test'] = {name: f"{model_output_dir}/{name}_output.json" for name in eval_datasets['test']}

        assert self.tokenizer, "Tokenizer must be provided if eval is on-the-fly"
        
        if do_eval_zero_step:
            # Initialize EvaluationCallback
            eval_callback = EvaluationCallback(eval_datasets, output_paths, self.tokenizer, eval_every_k_steps=save_every_steps)
            dummy_state = TrainerState()
            dummy_state.global_step = 0
            # Manually call the evaluation step
            eval_callback.on_step_end(args=None, state=dummy_state, control=None, model=self.model)


        if cfg['kd']:
            utils.log_w_wandb(logger=logger, message=f"----- Using KD Trainer ------")
            train_dl=custom_trainer_kd.get_train_dl(cfg,self.tokenizer)
            self.trainer = custom_trainer_kd.CustomTrainer(
                model=self.model,
                train_dl=train_dl,
                tokenizer=self.tokenizer,
                args=training_args,
                callbacks=[EvaluationCallback(eval_datasets, output_paths,self.tokenizer, eval_every_k_steps=save_every_steps)]
        )
        else:
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_lm_dataset["train"],
                eval_dataset=tokenized_lm_dataset["validation"],
                callbacks=[EvaluationCallback(eval_datasets, output_paths,self.tokenizer, eval_every_k_steps=save_every_steps)]
            )
        self.trainer.train()
        return self.trainer.model

def ft_model(cfg:Dict):
    
    tokenizer = get_tokenizer(cfg)
    lm_dataset = dp_train.get_tokenized_ds(cfg,tokenizer)
    trainer = NLPModelTrainer(cfg,tokenizer)
    trainer.train(cfg=cfg,tokenized_lm_dataset=lm_dataset)

def get_tasks_tokenized(cfg:Dict,tokenizer) -> DatasetDict:
    path_to_tokenize_ds=os.path.join(cfg['tokenizer_path'],'tokenized_datasets/QA')
    #if ds exists load it
    if os.path.exists(path_to_tokenize_ds+'/train.json') and os.path.exists(path_to_tokenize_ds+'/validation.json'):
        tokenized_dataset_train = Dataset.from_dict(json.load(open(path_to_tokenize_ds+'/train.json')))
        tokenized_dataset_validation = Dataset.from_dict(json.load(open(path_to_tokenize_ds+'/validation.json')))
        return DatasetDict({"train":tokenized_dataset_train,"validation":tokenized_dataset_validation})

    input_ids, attention_mask,labels = [],[],[] 

    for path in TASKS_DATA_PATHS:
        with open(path, 'r') as file:
            json_data = [json.loads(line) for line in file]

        if utils.is_debug_mode():
            json_data = json_data[:60]  # Take only the first 10 entries in debug mode

        # Tokenize the data
        tokenized_data = [tokenizer(data['problem'] + " solution:" + data['solution']) for data in json_data]
        input_ids+=[d['input_ids'] for d in tokenized_data]
        attention_mask+=[d['attention_mask'] for d in tokenized_data]
        labels+=[d['input_ids'] for d in tokenized_data]
        # Prepare the dataset
    tokenized_dataset = Dataset.from_dict({
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    })
    num_validation_samples = 20
    tokenized_dataset_validation = tokenized_dataset.select(range(num_validation_samples))
    tokenized_dataset_train = tokenized_dataset.select(range(num_validation_samples, len(tokenized_dataset)))
    #save tokenized datasets
    if not utils.is_debug_mode():
        os.makedirs(path_to_tokenize_ds, exist_ok=True)
        json.dump(tokenized_dataset_train.to_dict(), open(path_to_tokenize_ds+'/train.json', 'w'))
        json.dump(tokenized_dataset_validation.to_dict(), open(path_to_tokenize_ds+'/validation.json', 'w'))
    return DatasetDict({"train":tokenized_dataset_train,"validation":tokenized_dataset_validation})
        
        

def get_tokenizer(cfg:Dict):
    if not 'vanila' in cfg['tokenizer_path']:
        # load patch tokenizer:
        tokenizer = PatchTokenizer.load_model_from_scratch(path = os.path.join(cfg['tokenizer_path'],'patch_tokenizer.pkl') ,
                                                                    existing_tokenizer_name=cfg['original_tokenizer'])
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg['original_tokenizer'])
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune model with specified configuration.")
    parser.add_argument('--unfrozen', type=int, nargs='+', help='Specify layers to unfreeze, e.g., --unfrozen 1 2 will unfrozen first layer and two last') 
    parser.add_argument('--emb_method', type=str, help='Specify embedding method, e.g., --emb_method EXP_EMB')
    parser.add_argument('--dataset', type=str, help='dataset name')
    parser.add_argument('--lora', type=bool, default=False, help='use lora')
    return parser.parse_args()


if __name__ == "__main__":
    utils.set_seed(SEED)
    args = parse_args()
    
    cfg_tok=CONFIG_P_TOKENIZER
    cfg_ft=CONFIG_FT

    if args.unfrozen is not None:
        cfg_ft['unfrozen'] = tuple(args.unfrozen)
        print(f"unfrozen: {cfg_ft['unfrozen']}")

    if args.emb_method is not None:
        cfg_ft['emb_method'] = args.emb_method

    if args.dataset is not None:
        cfg_tok['target_corpus_name'] = args.dataset
        cfg_tok['analyze_ds_names'] = [args.dataset]
    if args.lora:
        cfg_ft['use_lora'] = True
        
    cfg = {**cfg_tok, **cfg_ft}
    utils.log_w_wandb(logger=logger, message=f"Starting fine-tuning model {cfg}")
    tokenizer_path=f"{SAVED_PATCH_TOKENIZER_PATH}/{utils.cfg_to_filename(cfg_tok)}" if cfg['emb_method']!=VANILA_MODEL else f"{SAVED_PATCH_TOKENIZER_PATH}/_vanila_{cfg['original_tokenizer']}_{cfg['target_corpus_name'][len('machelreid/m2d2_'):]}"
    utils.log_w_wandb(logger=logger, message=f"loading tokenizer from {tokenizer_path}")
    cfg['tokenizer_path']=tokenizer_path 
    ft_model(cfg)