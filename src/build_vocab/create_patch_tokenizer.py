import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.constants import *
from transformers import AutoTokenizer,AutoModelForCausalLM
import utils.utils_funcs as utils
from domain_selection import compare_datasets
from reduce_vocab import reduce_vocab
from build_vocab.add_ngrams import find_ngrams_to_add
import pickle
import os
from build_vocab.patch_tokenizer import PatchTokenizer
import build_vocab.handle_ds_pt as handle_ds_pt
import change_embeddings_in_model
import find_ngrams_to_add
utils.set_seed(SEED)


def pt_has_required_Files(directory_path, required_files= ['config.pkl', 'removed_tokens.pkl', 'added_ngrams.pkl', 'patch_tokenizer.pkl'])->bool:

    if not os.path.exists(directory_path):
        return False
    existing_files = set(os.listdir(directory_path))
    return all(file in existing_files for file in required_files)


def create_patch_tokenizer(cfg:Dict,debug_runs:bool=False):
    original_tokenizer = AutoTokenizer.from_pretrained(cfg['original_tokenizer'])
    utils.log_w_wandb(logger=logger,message=f"starting run_patch_tokenizer")
    print(f'for debug: {cfg["analyze_ds_names"]}')
    ds_names, config_names = zip(*[utils.get_ds_config_names(ds_name) for ds_name in cfg['analyze_ds_names']])
    utils.log_w_wandb(logger=logger,message=f"'target' corpus config: {cfg['target_corpus_name']}")
    utils.log_w_wandb(logger=logger,message=f"number of tokens to change: {cfg['num_to_add']}")
    
    tokenizer_dir_path = os.path.join(f"{REPO_PATH}/src/saved_patch_tokenizers_no_ngrams_new_logs", f'{utils.cfg_to_filename(cfg)}')
    # if path exists - return:
    if pt_has_required_Files(tokenizer_dir_path):
        utils.log_w_wandb(logger=logger,message=f"tokenizer already exists in {tokenizer_dir_path}")
        return
    else:
        os.makedirs(tokenizer_dir_path, exist_ok=True)
    
    with open(os.path.join(tokenizer_dir_path, 'config.pkl'), 'wb') as f:
        pickle.dump(cfg, f)

        
    if cfg['analyze_ds_names']==[]:
        ds_names,config_names = ([],[])
    runner_original_tokenizer = compare_datasets.AnalysisRunner(tokenizer=original_tokenizer, dataset_names=ds_names,n=0,k_most_ngrams=[],ds_configs=config_names)
    results_before_reduce=runner_original_tokenizer.run_analysis()

    # handle ds:
    target_ds = handle_ds_pt.get_target_ds(cfg)

    tokenized_target_ds = handle_ds_pt.get_tokenized_ds(cfg=cfg,ds=target_ds['train'],tokenizer=original_tokenizer)
    
    tokens_freqs = handle_ds_pt.get_tokens_freqs(cfg=cfg,tokenized_corpus=tokenized_target_ds,big_corpus_tokenized=None)
    
    # ---------- remove tokens:
    removed_tokens_file_path = os.path.join(tokenizer_dir_path, 'removed_tokens.pkl')
    if not os.path.exists(removed_tokens_file_path):
        removed_tokens = reduce_vocab(original_tokenizer, tokens_freqs, cfg=cfg)
    
        with open(removed_tokens_file_path, 'wb') as f:
            pickle.dump(removed_tokens, f)
    else:
        # Load existing removed tokens if needed
        with open(removed_tokens_file_path, 'rb') as f:
            removed_tokens = pickle.load(f)

    # update num to add to be not greater than the number of tokens that were removed
    cfg['num_to_add']=min(cfg['num_to_add'],len(removed_tokens))
    
    reduced_tokenizer=PatchTokenizer(existing_tokenizer_name=cfg['original_tokenizer'],removed_tokens=removed_tokens)
    
    #runner_original_tokenizer = compare_datasets.AnalysisRunner(tokenizer=reduced_tokenizer, dataset_names=ds_names,ds_configs=config_names)
    #results_after_reduce=runner_original_tokenizer.run_analysis()


    # ---------- add tokens:
    added_ngrams_file_path = os.path.join(tokenizer_dir_path, 'added_ngrams.pkl')
    if not os.path.exists(added_ngrams_file_path):
        tokenized_ds_after_rm = handle_ds_pt.get_tokenized_ds(cfg=cfg, ds=target_ds['train'], tokenizer=reduced_tokenizer)
        
        current_vocab = [token for token in original_tokenizer.get_vocab().keys() if token not in removed_tokens.keys()]
        print(f'### Adding Ngrams #######')
        added_ngrams = find_ngrams_to_add.main(tokenized_target_ds=tokenized_ds_after_rm,
                                               final_vocab_size=len(original_tokenizer.get_vocab()),current_vocab=current_vocab, cfg=cfg)
        with open(added_ngrams_file_path, 'wb') as f:
            pickle.dump(added_ngrams, f)
    else:
        # Load existing added ngrams if needed
        with open(added_ngrams_file_path, 'rb') as f:
            added_ngrams = pickle.load(f)
    assert len(added_ngrams) == len(removed_tokens), "Number of added ngrams should be equal to the number of removed tokens"
    new_tokenizer = PatchTokenizer(existing_tokenizer_name=cfg['original_tokenizer'],removed_tokens=removed_tokens,ngram_dict=added_ngrams)

    runner_new_tokenizer = compare_datasets.AnalysisRunner(tokenizer=new_tokenizer, dataset_names=ds_names,ds_configs=config_names)
    results_after_changes=runner_new_tokenizer.run_analysis()

        # -------- edit model:
    model = AutoModelForCausalLM.from_pretrained(cfg['original_tokenizer'])
    inputs_ids_convert=change_embeddings_in_model.get_modified_emb(removed_tokens=removed_tokens,added_ngrams=added_ngrams,
                                                                    model=model,tokenizer=new_tokenizer,base_path=tokenizer_dir_path)
    
    new_tokenizer.add_converter(inputs_ids_convert)
    
    new_tokenizer.save_model(path=os.path.join(tokenizer_dir_path, 'patch_tokenizer.pkl'))
    
    utils.log_w_wandb(logger=logger,message=f"saved tokenizer to {tokenizer_dir_path}")
    
    dataset_name, config_name = zip(*[utils.get_ds_config_names(ds_name) for ds_name in cfg['analyze_ds_names']])
    dataset_name, config_name = dataset_name, config_name


if __name__ == "__main__":
    cfg=CONFIG_P_TOKENIZER
    current_vocab_size=AutoTokenizer.from_pretrained(cfg['original_tokenizer']).vocab_size
    new_vocab_size = current_vocab_size

    domains = [
    "Natural_and_physical_sciences__Earth_sciences",
    "Culture_and_the_arts__Games_and_Toys",
    "physics.hist-ph",
    ]
    for domain in domains:
            domain= 'machelreid/m2d2_'+domain
            cfg['analyze_ds_names'] = [domain]
            cfg['target_corpus_name'] = domain
            print(f' ----------------- %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% -----------------')
            print(f'----------------- %%%%%%   running on {domain}   %%%%%%% -----------------')
            print(f' ----------------- %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% -----------------')
            print(f'running on {domain}')
            create_patch_tokenizer(cfg=CONFIG_P_TOKENIZER)