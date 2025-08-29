import os
import json
import pandas as pd

def calculate_saved_tokens_percentage(sentences):
    # Count total tokens if every list was separated
    separate_token_count = 0
    grouped_token_count = 0
    for sentence in sentences:
        for token in sentence:
            if isinstance(token, list):
                separate_token_count += len(token)
            else:
                separate_token_count += 1

        # Count total tokens in grouped structure
        grouped_token_count += len(sentence)

    saved_percentage = 100 * (1 - grouped_token_count / separate_token_count)

    # Round to 3 decimal places
    return round(saved_percentage, 3)

def process_files_in_folder(folder_path):
    results = []
    
    # Iterate through each file in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json") and "generated_data_output" in file_name:
            file_path = os.path.join(folder_path, file_name)
            step = file_name.split("_test_step_")[-1].split(".")[0]  # Extract the step number
            if not step.isdigit():
                step = file_name.split("_test_step_")[-1].split(".")[1].split("_")[-1]  # Extract the step number
            with open(file_path, "r") as file:
                data = json.load(file)
                outputs_as_tokens = data.get("outputs_as_tokens", [])
                total_percentage = calculate_saved_tokens_percentage(outputs_as_tokens)
                results.append({"step": int(step), "saved_tokens_percentage": total_percentage})
    
    # Convert results to DataFrame and save as CSV
    results_df = pd.DataFrame(results)
    results_df.sort_values("step", inplace=True)
    output_file = os.path.join(folder_path, "saved_tokens_generation.csv")
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


#main:
if __name__ == "__main__":
    # Specify the folder path containing the JSON files
    paths = [
        "./outputs/all_evals_outputs/llama_evals/hist/trained_models_0506/model_d82cfd39ed_sweep_firm-leaf-361_outputs",
    ]
    for folder in paths:
        process_files_in_folder(folder)
