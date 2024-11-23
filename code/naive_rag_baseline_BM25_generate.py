"""
This script is to get the oracle (upper bound) for the model performance.
"""

import os
import ast
import pdb
import json
import utils
import argparse
from tqdm import tqdm
from QwenGeneration import QwenGeneration

# add baseline_name to the parser
parser = argparse.ArgumentParser()
parser.add_argument("--baseline_type", type=str, default="oracle")
args = parser.parse_args()

if args.baseline_type == "oracle":
    key_to_ans = "Qwen_oracle_answer"
elif args.baseline_type == "zero_shot":
    key_to_ans = "Qwen_zero_shot_answer"

"""
LOAD DATA
"""
frames_file = "data/Qwen_Outputs/naive_rag_baseline_BM25_retrieve.jsonl"
with open(frames_file, "r") as f:
    frames_data = [json.loads(x.strip()) for x in f.readlines() if x.strip()]
print("Loaded frames_data.")

wiki_url_contents_file = "data/wikipedia/jsonl_output/wikipedia_filtered_url_to_content.json"
with open(wiki_url_contents_file, "r") as f:
    wiki_url_contents_dict = json.load(f)
print("Loaded wiki_url_contents_dict.")

"""
GENERATE RESPONSE
"""
qwen_model = QwenGeneration()
print("Qwen model loaded.")

naive_BM25_output_file = "data/Qwen_Outputs/naive_BM25_output.jsonl"
start_point = utils.get_start_point(naive_BM25_output_file)

with open(naive_BM25_output_file, "a") as f:
    for dict_item in tqdm(frames_data[start_point:]):
        context = utils.prepare_context(
            dict_item,
            wiki_url_contents_dict,
            key_to_links='naive_rag_retrieve_results'
            )
        query = dict_item["Prompt"]

        prompt = utils.oracle_prompt_template.format(
            context=context,
            question=query
            )

        response = qwen_model.get_response(prompt)
        grouth_truth_ans = dict_item["Answer"]

        print(f"Ground truth answer: {grouth_truth_ans}")
        print(f"Qwen response: {response}")

        dict_item["Qwen_naive_BM25_answer"] = response.strip()

        f.write(json.dumps(dict_item) + "\n")
