import os
import ast
import pdb
import json
import utils
import argparse
from tqdm import tqdm
from GPTGeneration import GPTGeneration

def main():
    frames_file = "data/frames_dataset_2_5_links_filtered.jsonl"
    with open(frames_file, "r") as f:
        frames_data = [json.loads(x.strip()) for x in f.readlines() if x.strip()]
    print("Loaded frames_data")

    generate_output = f"data/frames_dataset_2_5_links_filtered_extracted_queries_max_None.jsonl"

    processed_items = set()
    if os.path.exists(generate_output):
        with open(generate_output, "r") as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    processed_items.add(item["Prompt"])
                except:
                    continue

    """
    GENERATE RESPONSE
    """
    gpt_model = GPTGeneration(max_tokens=512)
    print("GPT4 model loaded.")
    print(f"Starting decompose query  with few-shot dynamic n.")

    remaining_frames = [item for item in frames_data if item["Prompt"] not in processed_items]
    print(f"Found {len(processed_items)} already processed items. {len(remaining_frames)} items remaining.")

    for dict_item in tqdm(remaining_frames):
        query = dict_item["Prompt"]
        prompt = utils.extract_query_dynamic_fewshot_prompt_template.format(
            question=query
        )
        
        valid = False
        attempts = 0
        while not valid:
            response, _, _ = gpt_model.get_response(prompt, judge=False, return_json=True)
            subq = response['atomic_questions']
            if len(subq) > 0:
                valid = True
            else:
                attempts += 1
                print(f"Attempt {attempts} failed. Retrying...")
                if attempts > 5:
                    pdb.set_trace()
        
        dict_item["extracted_subquery"] = subq
        
        with open(generate_output, "a") as f:
            f.write(json.dumps(dict_item) + "\n")
            f.flush()

if __name__ == "__main__":
    main()
