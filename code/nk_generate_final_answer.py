"""
This script is to get the oracle (upper bound) for the model performance.
"""

import os
import ast
import pdb
import json
import utils
from tqdm import tqdm
from QwenGeneration import QwenGeneration
import argparse

"""
LOAD DATA
"""

parser = argparse.ArgumentParser()
parser.add_argument(
    "--retriever",
    type=str,
    default="bm25",
    choices=["bm25", "dpr"],
    help="Retriever type to use (bm25, dpr)"
)
parser.add_argument(
    "--steps",
    type=int,
    default=5,
    help="Number of steps for n-step retrieval"
)
parser.add_argument(
    "--queries",
    type=int,
    default=5,
    help="Number of queries to generate at each step"
)
parser.add_argument(
    "--docs",
    type=int,
    default=10,
    help="Number of documents to retrieve at each step"
)
args = parser.parse_args()
nk_answer_file = f"data/Qwen_Outputs/nk_{args.retriever}_step_{args.steps}_queries_{args.queries}_docs_{args.docs}_answer.jsonl"
nk_file = f"data/Qwen_Outputs/nk_{args.retriever}_step_{args.steps}_queries_{args.queries}_docs_{args.docs}_output.jsonl"
with open(nk_file, "r") as f:
    nk_data = [json.loads(x.strip()) for x in f.readlines() if x.strip()]
print("Loaded nk_data.")

"""
GENERATE RESPONSE
"""

existed_ids = set()
if os.path.exists(nk_answer_file):
    print(f"{nk_answer_file} already exists")
    with open(nk_answer_file, 'r', encoding='utf-8') as f:
        nk_answer_ids = [json.loads(x.strip())['ID'] for x in f.readlines() if x.strip()]
        existed_ids = set(nk_answer_ids)
print(len(existed_ids))
qwen_model = QwenGeneration()
print("Qwen model loaded.")
with open(nk_answer_file, "a") as f:
    for dict_item in tqdm(nk_data):
        query_id = dict_item['ID']
        if query_id in existed_ids:
            # print("duplicated ID")
            continue
        context = dict_item['final_context']
        query = dict_item["Prompt"]

        prompt = utils.oracle_prompt_template.format(
            context=context,
            question=query
        )
        response = qwen_model.get_response(prompt)
        ground_truth_ans = dict_item["Answer"]

        print(f"Ground truth answer: {ground_truth_ans}")
        print(f"Qwen response: {response}")

        dict_item["Qwen_answer"] = response.strip()
        existed_ids.add(query_id)

        f.write(json.dumps(dict_item) + "\n")
