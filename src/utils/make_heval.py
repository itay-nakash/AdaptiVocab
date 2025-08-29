import os
import json
import random
import csv

# Folder paths
base_folder = "/data/home/itay.nakash/patch_tokenizer/outputs/h_eval_data"
domains = ["hist_science", "physical_science", "toys"]
# Each domain has these three files
model_files = {
    "av":       "_av_gens.json",
    "vanilla":  "_vanila.json",
    "vanilla_ft":"_vanila_ft.json"
}

# We want to compare these pairs -> 3 pairs per domain => 9 files total
pairs = [
    ("av", "vanilla_ft"),  # Pair1
    ("av", "vanilla"),     # Pair2
    ("vanilla", "vanilla_ft")  # Pair3
]

# Output folder
out_folder = "./data/splited_randomed_he"
os.makedirs(out_folder, exist_ok=True)

# We'll create 25 examples in each CSV file
N = 25

for domain in domains:
    for i,(m1, m2) in enumerate(pairs):
        # Load the two JSON files
        fpath1 = os.path.join(base_folder, domain, domain.split("_")[0] + model_files[m1])
        fpath2 = os.path.join(base_folder, domain, domain.split("_")[0] + model_files[m2])

        with open(fpath1, 'r', encoding='utf-8-sig') as f:
            data1 = json.load(f)["outputs"]  # list of 50 sentences
        with open(fpath2, 'r', encoding='utf-8-sig') as f:
            data2 = json.load(f)["outputs"]  # list of 50 sentences

        # We only need 25 from each for the CSV
        # (shuffle or just take first 25 -- here we show random)
        if i==0:
            data1 = data1[:N] # av
            data2 = data2[:N] # vanilla_ft
        if i==1:
            data1 = data1[N:2*N] # av
            data2 = data2[N:2*N] # vanilla
        if i==2:
            data1 = data1[N:2*N] # vanilla
            data2 = data2[N:2*N] # vanilla_ft

        # Randomly choose which model goes in column "model1" vs. "model2"
        rows = []
        for d1, d2 in zip(data1, data2):
            # 50-50 chance swap
            if random.random() < 0.5:
                row_m1, row_m2 = d1, d2
            else:
                row_m1, row_m2 = d2, d1

            rows.append({
                "domain": domain,
                "model1": row_m1,
                "model2": row_m2,
                "logic": "",
                "coherence": "",
                "linguistic_correctness": ""
            })

        # Write out to CSV
        out_name = f"{domain}_{m1}_vs_{m2}.csv"
        out_path = os.path.join(out_folder, out_name)
        with open(out_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
            fieldnames = ["domain", "model1", "model2", "logic", "coherence", "linguistic_correctness"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        print(f"Created: {out_path}")
