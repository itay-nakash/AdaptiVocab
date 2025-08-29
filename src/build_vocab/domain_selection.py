import pandas as pd
from datasets import load_dataset
from utils.constants import *

def stats_domains():
    results_df = pd.DataFrame(columns=['cfg_name', 'total_tokens', 'number_of_examples', 'number_of_filtered_examples','words_to_tokens_ratio'])

    #load current domains from dataset_summary.csv
    MACHELREID_M2D2_CONFIGS[len(results_df):]
    for cfg in MACHELREID_M2D2_CONFIGS:
        try:
            #if in the dataset_summary.csv dont run again
            if cfg in results_df['cfg_name'].values:
                continue
            # Load the dataset
            ds = load_dataset('machelreid/m2d2', cfg)

            # Assuming the data has a column 'text' that contains the text examples
            filtered_examples = [ex for ex in ds['train']['text'] if len(ex.split()) >= 40]

            # Compute total tokens and number of examples
            number_of_examples = len(filtered_examples)
            number_of_filtered_examples = len(filtered_examples)
            total_words = sum(len(ex.split()) for ex in filtered_examples)

            # Append results to the DataFrame
            results_df = results_df.append({
                'cfg_name': cfg,
                'total_words': total_words,
                'number_of_examples': number_of_examples, 
                'number_of_filtered_examples': number_of_filtered_examples,
                'words_to_tokens_ratio': total_words / number_of_filtered_examples
                
            }, ignore_index=True)

            # Save after processing each configuration
            results_df.to_csv('dataset_summary_new.csv', index=False)

        except Exception as e:
            print(f"Failed to process configuration {cfg} due to error: {e}")


def get_domain_text():
    cfg_list = [
    "math_l1",
    "math.RA",
    "math.GN",
    "math.OA",
    "Mathematics_and_logic__Fields_of_mathematics",
    "dg-ga",
    "solv-int",
    "History_and_events",
    "History_and_events__By_continent",
    "Natural_and_physical_sciences__Physical_sciences",
    "Culture_and_the_arts__Sports_and_Recreation",
    "astro-ph.EP",
    "Health_and_fitness__Public_health",
    "physics.atm-clus",
    "comp-gas",
    "Health_and_fitness__Health_science",
    "physics.atom-ph",
    "Culture_and_the_arts__The_arts_and_Entertainment",
    "hep-th",
    "Natural_and_physical_sciences__Earth_sciences",
    "cond-mat.quant-gas",
    "cs.FL",
    "Health_and_fitness__Self_care",
    "History_and_events__By_region",
    "Health_and_fitness__Exercise",
    "Philosophy_and_thinking__Philosophy"
]

    for domain in cfg_list:
        ds=load_dataset('machelreid/m2d2', domain)
        to_save=[]
        print(f'------ processing {domain} --------')
        for exmpl in ds['train']['text']:
            if len(to_save)>10_000:
                break
            if len(exmpl.split())<40:
                continue
            to_save.append(exmpl)
        # save in a format easy to load as dataset:
        print(f'------ saving {domain} --------')
        with open(f'{REPO_PATH}/for_nitay_exampels/{domain}.txt','w') as f:
            f.write('\n'.join(to_save))
        to_save=[]

if __name__ == "__main__":
    get_domain_text()