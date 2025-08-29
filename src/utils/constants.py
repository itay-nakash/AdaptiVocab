from typing import List, Tuple, Dict, Any, Union, Callable, Optional, Collection
from collections import defaultdict
from pathlib import Path
from pandas import DataFrame
from numpy import ndarray
import logging 
import torch
import sys
import os

#logger:
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#wandb projct:
WANDB_PROJECT_NAME='patch_tokenizer_kd2'


# General
BASE_PATH='/data/home/itay.nakash/patch_tokenizer/'
SEED = 42
EPSILON = 0.0000001
LENGTH_THRESHOLD = 1000 # maximum length of a sentence for the stats
K_FOR_NGRAMS = 4 
EXP_EMB='exp_emb'
MEAN_EMB='mean_emb'
VANILA_TOKENIZER_NAME='vanila_tokenizer'
NOT_FOUND='not_found'
MIN_EXAMPLE_LEN=50

#Paths:
CUR_DIR = os.path.dirname(__file__)
PARTENT_DIR = os.path.dirname(CUR_DIR)
PARENT_2_DIR = os.path.dirname(PARTENT_DIR)
REPO_PATH=f'/data/home/itay.nakash/patch_tokenizer/'
SAVED_PATCH_TOKENIZER_PATH=f'{REPO_PATH}/src/saved_patch_tokenizers_ngram_k_analysis'
DATA_PATH=f'{REPO_PATH}/data'
PROCESSED_DATA_PATH=f'{DATA_PATH}/processed_datasets'
DATASET_STATS_PATH=f'{DATA_PATH}/ds_stats'
TOKENIZED_DS='tokenized_datasets'
HASH_FILE_PATH=f'{REPO_PATH}/src/utils/hashes.json'
TRAINED_MODELS_PATH=f'{REPO_PATH}/trained_models_0506'
EVAL_OUTPUTS_PATH=f'{REPO_PATH}/outputs/all_evals_outputs'
PATH_TO_TASKS_DATA=f'{REPO_PATH}/evaluate/QA'
PATH_TO_CRED='/data/home/itay.nakash/patch_tokenizer/gemma-nlp-422508-c4d4ce542552-gilat.json'
TASKS_DATA_PATHS = [
    f'{REPO_PATH}/evaluate/QA/ARC/data/train.jsonl',
    f'{REPO_PATH}/evaluate/QA/ECQA/data/train.jsonl',
    f'{REPO_PATH}/evaluate/QA/OpenbookQA/data/train.jsonl',
    f'{REPO_PATH}/evaluate/QA/QASC/data/train.jsonl',
    # Add more paths as needed
]


# New Embeddings:
CUSTOM_BPE='custom_bpe'
RANDOM_EMB='random_emb'
EXP_EMB='exp_emb'
MEAN_EMB='mean_emb'
VANILA_MODEL='vanila'
MEAN_EMBEDDING='mean'
DECREASING_EXP='dec_exp'
INCREASING_EXP='inc_exp'


# 'target_corpus_name': 'machelreid/m2d2_Culture_and_the_arts__The_arts_and_Entertainment',
CONFIG_P_TOKENIZER = {
        #'original_tokenizer': 'meta-llama/Llama-2-7b-hf',
        'original_tokenizer': 'mistralai/Mistral-7B-v0.3',
       # 'original_tokenizer': 'mistralai/Mistral-7B-Instruct-v0.2',
        'target_corpus_name': 'machelreid/m2d2_Natural_and_physical_sciences__Earth_sciences',
      #  'target_corpus_name': 'machelreid/m2d2_Health_and_fitness__Self_care',
        'num_to_add': 10000,
        'num_to_remove':10000,
        'analyze_ds_names':['machelreid/m2d2_Natural_and_physical_sciences__Earth_sciences'],
      #  'analyze_ds_names':['machelreid/m2d2_Health_and_fitness__Self_care'],
        'ngram_k':3,
#       'vocab_size': 'original',
      #  'alpha':0,
      #  'big_corpus_name': 'None',
    }

CONFIG_FT = {
        'num_epochs': 1,
        'use_lora': False,
        'unfrozen': (1,1),
        'lr': 5e-5,
        'weight_decay': 0.01,
        'warmup_ratio': 0.0,
        'batch_size': 1,
        'gradient_accumulation_steps': 32,
        'torch_dtype': torch.float,
        'example_to_train': sys.maxsize,
        'example_to_val': 10_000,
        'emb_method': EXP_EMB,   
        'kd': False,
    #    'debug_tests':7,
    #    'move_emb':-1,
        #'kd_temperature':0.5,
    }

MAX_VALIDATION_SIZE=40000


# Reduce_vocab:
FREQ_TRIM_THRESHOLD = 50
QA_SAVED_TOKENS = ['Your','choice','Question','Options','Format',"▁'.",'Which']
SAVED_TOKENS = ['<pad>', '<unk>', '<s>', '</s>'] + QA_SAVED_TOKENS

# Add_ngrams:
MIN_FREQ_AFFECTED=2


# Model:
LOSS_IGNORE_ID = -100

#training:
EMB_HEAD = 'emb,head'
EMB_HEAD_LAST= 'emb,head,last'
SAVE_EVERY_STEPS=100


# tokenization:

UNIQE_TOKEN_TO_ADD='정'

# test generation:

EXAMPELS_TO_VAL = 10_000
MIN_LEN_INPUT=15

PROMPT_LEN_LIST=[27, 24, 17, 31, 28, 29, 27, 25, 25, 25, 17, 20, 23, 28, 23, 26, 21, 30, 23, 20, 
 31, 22, 32, 31, 16, 27, 29, 21, 17, 17, 24, 30, 26, 26, 25, 19, 22, 20, 28, 29, 
 30, 31, 20, 16, 26, 25, 18, 32, 16, 22, 31, 16, 25, 27, 32, 31, 30, 19, 17, 21, 
 23, 20, 32, 20, 24, 20, 32, 21, 22, 31, 31, 20, 28, 17, 19, 30, 18, 18, 28, 29, 
 17, 32, 25, 23, 23, 25, 19, 17, 25, 25, 21, 26, 30, 28, 28, 31, 26, 23, 25, 19, 
 25, 22, 18, 27, 21, 29, 31, 32, 16, 27, 29, 16, 23, 20, 28, 23, 27, 29, 26, 16, 
 31, 21, 30, 25, 23, 28, 20, 18, 21, 20, 18, 18, 20, 16, 19, 26, 20, 16, 22, 16, 
 18, 17, 30, 23, 29, 31, 23, 25, 26, 20, 16, 26, 17, 31, 23, 28, 29, 19, 20, 25, 
 27, 19, 20, 20, 21, 23, 20, 27, 19, 19, 31, 24, 22, 30, 22, 32, 27, 29, 28, 31, 
 25, 20, 21, 17, 18, 32, 17, 17, 31, 30, 29, 16, 18, 19, 24, 17, 23, 28, 32, 16, 
 30, 30, 20, 24, 25, 22, 19, 26, 32, 16, 22, 27, 22, 18, 24, 27, 22, 31, 25, 31, 
 17, 16, 21, 22, 30, 31, 21, 28, 32, 21, 30, 27, 18, 17, 22, 31, 29, 32, 22, 21, 
 16, 16, 20, 16, 19, 23, 31, 19, 22, 21, 27, 21, 17, 25, 17, 30, 16, 21, 30, 32, 
 16, 19, 22, 22, 28, 27, 24, 18, 16, 23, 18, 19, 25, 32, 24, 16, 32, 20, 20, 22, 
 17, 17, 22, 30, 32, 23, 19, 21, 31, 20, 26, 20, 21, 18, 22, 30, 18, 22, 23, 31, 
 17, 27, 21, 24, 21, 16, 17, 16, 30, 21, 28, 20, 16, 20, 23, 28, 28, 31, 32, 21, 
 22, 21, 24, 21, 16, 32, 17, 24, 31, 22, 22, 30, 31, 24, 20, 29, 31, 23, 29, 17, 
 28, 18, 24, 19, 26, 30, 18, 25, 19, 22, 27, 16, 17, 23, 21, 17, 16, 31, 31, 29, 
 23, 21, 28, 30, 23, 23, 25, 17, 21, 20, 19, 31, 31, 18, 28, 29, 19, 30, 23, 27, 
 20, 23, 18, 17, 16, 27, 19, 21, 23, 27, 18, 26, 18, 17, 16, 23, 24, 20, 19, 22, 
 29, 21, 31, 24, 21, 26, 30, 25, 23, 24, 20, 26, 26, 17, 23, 17, 31, 27, 27, 22, 
 32, 30, 29, 17, 32, 21, 16, 21, 30, 24, 24, 18, 29, 26, 16, 19, 24, 23, 25, 28, 
 18, 21, 18, 28, 22, 24, 22, 29, 21, 17, 32, 20, 30, 19, 17, 31, 20, 20, 23, 16, 
 19, 22, 23, 23, 30, 23, 20, 21, 21, 30, 18, 27, 25, 24, 23, 21, 19, 32, 26, 20, 
 24, 27, 22, 19, 24, 26, 23, 19,19, 22, 23, 23, 30, 23, 20, 21, 21, 30, 18, 27, 25, 24, 23, 21, 19, 32, 26, 20, 
 24, 27, 22, 19, 24, 26, 23, 19, 19, 20, 20, 21, 23, 20, 27, 19, 19, 31, 24, 22, 30, 22, 32, 27, 29, 28, 31, 
 25, 20]


# compare ds:
MACHELREID_M2D2_CONFIGS=['Art', 'Culture_and_the_arts', 'Culture_and_the_arts__Culture_and_Humanities', 'Culture_and_the_arts__Games_and_Toys', 'Culture_and_the_arts__Mass_media', 'Culture_and_the_arts__Performing_arts', 'Culture_and_the_arts__Sports_and_Recreation', 'Culture_and_the_arts__The_arts_and_Entertainment', 'Culture_and_the_arts__Visual_arts', 'General_referece', 'General_referece__Further_research_tools_and_topics', 'General_referece__Reference_works', 'Health_and_fitness', 'Health_and_fitness__Exercise', 'Health_and_fitness__Health_science', 'Health_and_fitness__Human_medicine', 'Health_and_fitness__Nutrition', 'Health_and_fitness__Public_health', 'Health_and_fitness__Self_care', 'History_and_events', 'History_and_events__By_continent', 'History_and_events__By_period', 'History_and_events__By_region', 'Human_activites', 'Human_activites__Human_activities', 'Human_activites__Impact_of_human_activity', 'Mathematics_and_logic', 'Mathematics_and_logic__Fields_of_mathematics', 'Mathematics_and_logic__Logic', 'Mathematics_and_logic__Mathematics', 'Natural_and_physical_sciences', 'Natural_and_physical_sciences__Biology', 'Natural_and_physical_sciences__Earth_sciences', 'Natural_and_physical_sciences__Nature', 'Natural_and_physical_sciences__Physical_sciences', 'Philosophy', 'Philosophy_and_thinking', 'Philosophy_and_thinking__Philosophy', 'Philosophy_and_thinking__Thinking', 'Religion_and_belief_systems', 'Religion_and_belief_systems__Allah', 'Religion_and_belief_systems__Belief_systems', 'Religion_and_belief_systems__Major_beliefs_of_the_world', 'Society_and_social_sciences', 'Society_and_social_sciences__Social_sciences', 'Society_and_social_sciences__Society', 'Technology_and_applied_sciences', 'Technology_and_applied_sciences__Agriculture', 'Technology_and_applied_sciences__Computing', 'Technology_and_applied_sciences__Engineering', 'Technology_and_applied_sciences__Transport', 'alg-geom', 'ao-sci', 'astro-ph', 'astro-ph.CO', 'astro-ph.EP', 'astro-ph.GA', 'astro-ph.HE', 'astro-ph.IM', 'astro-ph.SR', 'astro-ph_l1', 'atom-ph', 'bayes-an', 'chao-dyn', 'chem-ph', 'cmp-lg', 'comp-gas', 'cond-mat', 'cond-mat.dis-nn', 'cond-mat.mes-hall', 'cond-mat.mtrl-sci', 'cond-mat.other', 'cond-mat.quant-gas', 'cond-mat.soft', 'cond-mat.stat-mech', 'cond-mat.str-el', 'cond-mat.supr-con', 'cond-mat_l1', 'cs.AI', 'cs.AR', 'cs.CC', 'cs.CE', 'cs.CG', 'cs.CL', 'cs.CR', 'cs.CV', 'cs.CY', 'cs.DB', 'cs.DC', 'cs.DL', 'cs.DM', 'cs.DS', 'cs.ET', 'cs.FL', 'cs.GL', 'cs.GR', 'cs.GT', 'cs.HC', 'cs.IR', 'cs.IT', 'cs.LG', 'cs.LO', 'cs.MA', 'cs.MM', 'cs.MS', 'cs.NA', 'cs.NE', 'cs.NI', 'cs.OH', 'cs.OS', 'cs.PF', 'cs.PL', 'cs.RO', 'cs.SC', 'cs.SD', 'cs.SE', 'cs.SI', 'cs.SY', 'cs_l1', 'dg-ga', 'econ.EM', 'econ.GN', 'econ.TH', 'econ_l1', 'eess.AS', 'eess.IV', 'eess.SP', 'eess.SY', 'eess_l1', 'eval_sets', 'funct-an', 'gr-qc', 'hep-ex', 'hep-lat', 'hep-ph', 'hep-th', 'math-ph', 'math.AC', 'math.AG', 'math.AP', 'math.AT', 'math.CA', 'math.CO', 'math.CT', 'math.CV', 'math.DG', 'math.DS', 'math.FA', 'math.GM', 'math.GN', 'math.GR', 'math.GT', 'math.HO', 'math.IT', 'math.KT', 'math.LO', 'math.MG', 'math.MP', 'math.NA', 'math.NT', 'math.OA', 'math.OC', 'math.PR', 'math.QA', 'math.RA', 'math.RT', 'math.SG', 'math.SP', 'math.ST', 'math_l1', 'mtrl-th', 'nlin.AO', 'nlin.CD', 'nlin.CG', 'nlin.PS', 'nlin.SI', 'nlin_l1', 'nucl-ex', 'nucl-th', 'only_text', 'only_txt2', 'patt-sol', 'physics.acc-ph', 'physics.ao-ph', 'physics.app-ph', 'physics.atm-clus', 'physics.atom-ph', 'physics.bio-ph', 'physics.chem-ph', 'physics.class-ph', 'physics.comp-ph', 'physics.data-an', 'physics.ed-ph', 'physics.flu-dyn', 'physics.gen-ph', 'physics.geo-ph', 'physics.hist-ph', 'physics.ins-det', 'physics.med-ph', 'physics.optics', 'physics.plasm-ph', 'physics.pop-ph', 'physics.soc-ph', 'physics.space-ph', 'physics_l1', 'plasm-ph', 'q-alg', 'q-bio', 'q-bio.BM', 'q-bio.CB', 'q-bio.GN', 'q-bio.MN', 'q-bio.NC', 'q-bio.OT', 'q-bio.PE', 'q-bio.QM', 'q-bio.SC', 'q-bio.TO', 'q-bio_l1', 'q-fin.CP', 'q-fin.EC', 'q-fin.GN', 'q-fin.MF', 'q-fin.PM', 'q-fin.PR', 'q-fin.RM', 'q-fin.ST', 'q-fin.TR', 'q-fin_l1', 'quant-ph', 'solv-int', 'stat.AP', 'stat.CO', 'stat.ME', 'stat.ML', 'stat.OT', 'stat.TH', 'stat_l1', 'supr-con']


SYS_PROMPT_NON_INSTRUCT=""" You will receive multiple-choice questions presented as follows: 'Question: '. 
Your task is to respond with a detailed explanation and then provide the correct answer. 
Format your answer like this, substituting <YOUR_ANSWER> with the selected option from the provided choices: 
'The answer is: <YOUR_ANSWER>.'\n\n
"""
RESPONSE_PREFIX="\n\n Response: Let's analyze this step by step.\n"
SYS_PROMPT="You will be provided with multiple choice questions, Your task is to respond with an explanation and then write the correct answer in this format: 'The answer is: <YOUR_ANSWER>', where '<YOUR_ANSWER>' is the selected option from the provided choices."

FEW_SHOT_PROMPT_ARC= SYS_PROMPT_NON_INSTRUCT + """ Here a few exampels: Question:\n Which factor will most likely cause a person to develop a fever?\nOptions:\na leg muscle relaxing after exercise\na bacterial population in the bloodstream\nseveral viral particles on the skin\ncarbohydrates being digested in the stomach. \n\n
 Response: Let's analyze this step by step.\n The answer is a bacterial population in the bloodstream.
 Question:\n Lichens are symbiotic organisms made of green algae and fungi. What do the green algae supply to the fungi in this symbiotic relationship?\nOptions:\ncarbon dioxide\nfood\nprotection\nwater
 Response: Let's analyze this step by step.\n The answer is food
 Question:\n When a switch is used in an electrical circuit, the switch can\nOptions:\ncause the charge to build.\nincrease and decrease the voltage.\ncause the current to change direction.\nstop and start the flow of current.
 Response: Let's analyze this step by step.\n The answer is stop and start the flow of current.
  Now here is your qustion, answer in the same format presented in this prompt. 
 Question: \n  """
                                      
FEW_SHOT_PROMPT_ECQA= SYS_PROMPT_NON_INSTRUCT + """ Here a few exampels: Question:\n What might a person see at the scene of a brutal killing?\nOptions:\nbloody mess\npleasure\nbeing imprisoned\nfeeling of guilt\ncake  \n\n
 Response: Let's analyze this step by step.\n Bloody mess is covered or stained with blood.\nA person might see a bloody mess at the scene of a brutal killing.\nThe answer is bloody mess
 Question:\n If you want to set a romantic atmosphere you might light a candle where?\nOptions:\ndimly lit room\nsynagogue\nbedroom\nbirthday cake\nroses  \n\n
 Response: Let's analyze this step by step.\n A romantic atmosphere can be set in bedroom\nBedroom is a place where one sleeps\nCandles can be lit in a bedroom.\nThe answer is bedroom
 Question:\n What might the result of unwanted flirting be?\nOptions:\nattraction\nproblems\nthe gallows\nbeing slapped\ncuriosity  \n\n
 Response: Let's analyze this step by step.\n Person can be slapped if he does unwanted flirting to someone.\nThe answer is being slapped
 Now here is your qustion, answer in the same format presented in this prompt. 
 Question:\n """

FEW_SHOT_PROMPT_OPENBOOK= SYS_PROMPT_NON_INSTRUCT + """ Here a few exampels: Question:\n The sun is responsible for\nOptions:\npuppies learning new tricks\nchildren growing up and getting old\nflowers wilting in a vase\nplants sprouting, blooming and wilting  \n\n
 Response: The answer is plants sprouting, blooming and wilting
 Question:\n  When standing miles away from Mount Rushmore\nOptions:\nthe mountains seem very close\nthe mountains are boring\nthe mountains look the same as from up close\nthe mountains seem smaller than in photographs  \n\n
 Response: The answer is the mountains seem smaller than in photographs
 Question:\n  When food is reduced in the stomach\nOptions:\nthe mind needs time to digest\ntake a second to digest what i said\nnutrients are being deconstructed\nreader's digest is a body of works  \n\n
 Response: The answer is nutrients are being deconstructed
 Now here is your qustion, answer in the same format presented in this prompt. 
 Question:\n  """                                

FEW_SHOT_PROMPT_QASC= SYS_PROMPT_NON_INSTRUCT +""" Here a few exampels: Question:\n What type of water formation is formed by clouds?\nOptions:\npearls\nstreams\nshells\ndiamonds\nrain\nbeads\ncooled\nliquid  \n\n
 Response: beads of water are formed by water vapor condensing\nClouds are made of water vapor.\nBeads of water can be formed by clouds.\nThe answer is beads
 Question:\n Where do beads of water come from?\nOptions:\ntoo much water\nunderground systems\nwhen the water is too cold\nwater spills\nvapor turning into a liquid\nwarm air moving into cold air\nat the peak of a mountain\nto another location like underground  \n\n
 Response: beads of water are formed by water vapor condensing\nCondensation is the change of water vapor to a liquid.\nVapor turning into a liquid leaves behind beads of water\nThe answer is Vapor turning into a liquid
 Question:\n What forms beads of water?\nOptions:\nnecklaces.\nsteam.\nglass beads .\na wave\ntiny\na solute\nrain\nbracelets.  \n\n
 Response: beads of water are formed by water vapor condensing\nAn example of water vapor is steam.\nSteam forms beads of water.\nThe answer is Steam.
 Now here is your qustion, answer in the same format presented in this prompt. 
 Question:\n """
MSG_ARC = [
        {"role": "user", "content": SYS_PROMPT+" Which factor will most likely cause a person to develop a fever?\nOptions:\na leg muscle relaxing after exercise\na bacterial population in the bloodstream\nseveral viral particles on the skin\ncarbohydrates being digested in the stomach"},
        {"role": "assistant", "content": "The answer is a bacterial population in the bloodstream"},
        {"role": "user", "content": "Lichens are symbiotic organisms made of green algae and fungi. What do the green algae supply to the fungi in this symbiotic relationship?\nOptions:\ncarbon dioxide\nfood\nprotection\nwater"},
        {"role": "assistant", "content": "The answer is food"},
        {"role": "user", "content": "When a switch is used in an electrical circuit, the switch can\nOptions:\ncause the charge to build.\nincrease and decrease the voltage.\ncause the current to change direction.\nstop and start the flow of current."},
        {"role": "assistant", "content": "The answer is stop and start the flow of current."}
    ]
MSG_ECQA= [
        {"role": "user", "content": SYS_PROMPT+" What might a person see at the scene of a brutal killing?\nOptions:\nbloody mess\npleasure\nbeing imprisoned\nfeeling of guilt\ncake"},
        {"role": "assistant", "content": "Bloody mess is covered or stained with blood.\nA person might see a bloody mess at the scene of a brutal killing.\nThe answer is bloody mess"},
        {"role": "user", "content": "If you want to set a romantic atmosphere you might light a candle where?\nOptions:\ndimly lit room\nsynagogue\nbedroom\nbirthday cake\nroses"},
        {"role": "assistant", "content": "A romantic atmosphere can be set in bedroom\nBedroom is a place where one sleeps\nCandles can be lit in a bedroom.\nThe answer is bedroom"},
        {"role": "user", "content": "What might the result of unwanted flirting be?\nOptions:\nattraction\nproblems\nthe gallows\nbeing slapped\ncuriosity"},
        {"role": "assistant", "content": "Person can be slapped if he does unwanted flirting to someone.\nThe answer is being slapped"}
]

MSG_OPENBOOK= [
        {"role": "user", "content": SYS_PROMPT+" The sun is responsible for\nOptions:\npuppies learning new tricks\nchildren growing up and getting old\nflowers wilting in a vase\nplants sprouting, blooming and wilting"},
        {"role": "assistant", "content": "The answer is plants sprouting, blooming and wilting"},
        {"role": "user", "content": "When standing miles away from Mount Rushmore\nOptions:\nthe mountains seem very close\nthe mountains are boring\nthe mountains look the same as from up close\nthe mountains seem smaller than in photographs"},
        {"role": "assistant", "content": "The answer is the mountains seem smaller than in photographs"},
        {"role": "user", "content": "When food is reduced in the stomach\nOptions:\nthe mind needs time to digest\ntake a second to digest what i said\nnutrients are being deconstructed\nreader's digest is a body of works"},
        {"role": "assistant", "content": "The answer is nutrients are being deconstructed"}
]

MSG_QASC= [
        {"role": "user", "content": SYS_PROMPT+" What type of water formation is formed by clouds?\nOptions:\npearls\nstreams\nshells\ndiamonds\nrain\nbeads\ncooled\nliquid", "solution": "beads of water are formed by water vapor condensing\nClouds are made of water vapor.\nBeads of water can be formed by clouds.\nThe answer is beads"},
        {"role": "assistant", "content": "beads of water are formed by water vapor condensing\nClouds are made of water vapor.\nBeads of water can be formed by clouds.\nThe answer is beads"},
        
        {"role": "user", "content": "Where do beads of water come from?\nOptions:\ntoo much water\nunderground systems\nwhen the water is too cold\nwater spills\nvapor turning into a liquid\nwarm air moving into cold air\nat the peak of a mountain\nto another location like underground"},
        {"role": "assistant", "content": "beads of water are formed by water vapor condensing\nCondensation is the change of water vapor to a liquid.\nVapor turning into a liquid leaves behind beads of water\nThe answer is Vapor turning into a liquid"},
        
        {"role": "user", "content": "What forms beads of water?\nOptions:\nnecklaces.\nsteam.\nglass beads .\na wave\ntiny\na solute\nrain\nbracelets."},
        {"role": "assistant", "content": "beads of water are formed by water vapor condensing\nAn example of water vapor is steam.\nSteam forms beads of water.\nThe answer is Steam."}
]

PROMPT_MY_TASKS_PHYSICS=(f'"Please read the provided context and question carefully, then choose the best answer from the options. Provide the letter of the correct option and the option itself as your answer.\n"'
    'Example 1:\n'
    'Context: "<SOME CONTEXT>"\n'
    'Question: "Which of these locations is furthest south?"\n'
    'Options: \n'
    'A: Dagslett\n'
    'B: Peja\n'
    'C: Ivaylovgrad\n'
    'D: Spikkestad\n'
    'Answer: C, Ivaylovgrad\n'
    ''
    'Example 2:\n'
    'Context: "<SOME CONTEXT>"\n'
    'Question: "Which location was once considered as a potential site for an airport after World War II?"\n'
    'Options: \n'
    'A: Ivaylovgrad \n'
    'B: Lyutitsa\n'
    'C: Gornje Hrasno\n'
    'D: Ekebergsletta\n'
    'Answer: D, Ekebergsletta\n')

PROMPT_MY_TASKS_PHYSICS_NO_CONTEXT=(f'Please read the question carefully, then choose the best answer from the options. Provide the letter of the correct option and the option itself as your answer.\n'
    'Example 1:\n'
    'Question: "Which of these locations is furthest south?"\n'
    'Options: \n'
    'A: Dagslett\n'
    'B: Peja\n'
    'C: Ivaylovgrad\n'
    'D: Spikkestad\n'
    'Answer: C, Ivaylovgrad\n'
    ''
    'Example 2:\n'
    'Question: "Which location was once considered as a potential site for an airport after World War II?"\n'
    'Options: \n'
    'A: Ivaylovgrad \n'
    'B: Lyutitsa\n'
    'C: Gornje Hrasno\n'
    'D: Ekebergsletta\n'
    'Answer: D, Ekebergsletta\n')

INST_DICT={'arc':MSG_ARC,'ecqa':MSG_ECQA,'openbookqa':MSG_OPENBOOK,'qasc':MSG_QASC}

NON_INST_DICT={'arc':FEW_SHOT_PROMPT_ARC,'ecqa':FEW_SHOT_PROMPT_ECQA,'openbookqa':FEW_SHOT_PROMPT_OPENBOOK,'qasc':FEW_SHOT_PROMPT_QASC}

ANSWER_SHORTLY_SUFFIX = 'Answer with just one word, a number, a single number between 1 to 5.'
ASPECT_PROMPTS = {
"logic": (f"Please evaluate the logical consistency of the generated text below which continues the provided context. The text might end abruptly due to length limits; disregard any such truncation in your evaluation. Logical consistency measures how well the text adheres to logical principles, ensuring that arguments or narratives do not contradict themselves and remain rational throughout. Rate the logical consistency using a 5-point Likert scale defined as follows:"
"\n1: The text contains multiple contradictions or illogical statements that disrupt understanding."
"\n2: The text occasionally presents contradictions or illogical ideas that hinder comprehension."
"\n3: The text is generally logically consistent, with minor inconsistencies that do not majorly impact the overall logic."
"\n4: The text is mostly logically consistent, with all statements and ideas well-supported and rational."
"\n5: The text is completely logically consistent, with no contradictions, maintaining a rational and coherent argument or narrative from start to finish."
"\nUSER:\n During the Soviet period, the village was a part of the Askeran District of the Nagorno-Karabakh Autonomous Region, where the local economy inexplicably relied on exporting software technology to ancient civilizations through time travel initiatives."
"\nGENERATED 1\n USER: During the Soviet period, the village was a part of the Askeran District of the Nagorno-Karabakh Autonomous Region, where the local economy inexplicably relied on exporting software technology to ancient civilizations"
"\nGENERATED 3\n USER: During the Soviet period, the village was a part of the Askeran District of the Nagorno-Karabakh Autonomous Region, and it was known for its advancements in agricultural technology, although the region"
"\nUSER:\n Autonomous Region, known for its rich cultural heritage and traditional crafts. The villagers excelled in carpet weaving and pottery, which were significant contributors to their economy and played a vital role in preserving their cultura"
"\nGENERATED 5\n"
"Now evaluate the following generated sentence:"),

"coherence": (f"Please evaluate the coherency of the generated text below which continues the provided context. The text might end abruptly due to length limits; disregard any such truncation in your evaluation. Coherency measures how logically connected and consistent the text is across its entirety. Rate the coherency using a 5-point Likert scale defined as follows:"
"\n1: The text is very disjointed and lacks any logical connection between ideas."
"\n2: The text has minimal logical connection, with frequent disjointed or unrelated ideas."
"\n3: The text is somewhat coherent with occasional lapses in logical connection."
"\n4: The text is mostly coherent, with a clear logical flow of ideas throughout."
"\n5: The text is highly coherent, with all ideas logically connected and flowing seamlessly from start to finish."
"\nUSER:\n During the Soviet period, the village was a part of the Askeran District of the Nagorno-Karabakh Autonomous Region, where calendar market blue, the cat dances nightly, and river "
"\nGENERATED 1\n USER: During the Soviet period, the village was a part of the Askeran District of the Nagorno-Karabakh Autonomous Region, where the local economy inexplicably relied on exporting software technology to ancient civilizations"
"\nGENERATED 3\n USER: During the Soviet period, the village was a part of the Askeran District of the Nagorno-Karabakh Autonomous Region, where despite the challenging mountainous terrain, it was known for both its agricultural produce and a small, yet thriving tech startup scene, which seemed unusual given the remote location but was supported by local university graduates."
"\nUSER:\n Autonomous Region, known for its rich cultural heritage and traditional crafts. The villagers excelled in carpet weaving and pottery, which were significant contributors to their economy and played a vital role in preserving their cultura"
"\nGENERATED 5\n"
"Now evaluate the following generated sentence:"),

"linguistic_acceptability": (f"Please evaluate the linguistic acceptability of the generated text below which continues the provided context. The text might end abruptly due to length limits; disregard any such truncation in your evaluation. Linguistic acceptability measures how well the text adheres to the norms of grammar, syntax, and usage, ensuring that the language is correctly and appropriately used throughout. Rate the linguistic acceptability using a 5-point Likert scale defined as follows:"
"\n1: The text frequently uses incorrect grammar, syntax, or inappropriate language that severely disrupts understanding."
"\n2: The text often presents incorrect grammar or syntax, which hinders comprehension and distracts from the content."
"\n3: The text is generally linguistically acceptable, with occasional grammatical or syntactical errors that do not majorly impact readability."
"\n4: The text is mostly linguistically acceptable, with correct use of grammar and syntax, and only minor errors."
"\n5: The text is completely linguistically acceptable, with no grammatical or syntactical errors, using appropriate and precise language from start to finish."
"\n USER: \n During the Soviet period, the village was a part of the Askeran District of the Nagorno-Karabakh Autonomous Oblast, so roads wasn't paved and nobody didn't pay no mind to what law says about where your sheep"
"\n GENERATED 1\n USER: During the Soviet period, the village was a part of the Askeran District of the Nagorno-Karabakh Autonomous Region, where the local economy inexplicably relied on exporting software "
"\n USER: During the Soviet period, the village was a part of the Askeran District of the Nagorno-Karabakh Autonomous Oblast, where it seen much changes over time. Roads was mostly dirt, and many homes didn't have the"
"\n GENERATED 3\n USER: During the Soviet period, the village was a part of the Askeran District of the Nagorno-Karabakh Autonomous Oblast, where it flourished under the administrative oversight that prioritized "
"\nGENERATED 5\n"
"Now evaluate the following generated sentence:\n USER: \n")
}



# KD config:

SAME_AS_TEACHER_THRESHOLD=0.05

