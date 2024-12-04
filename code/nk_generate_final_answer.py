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
nk_answer_file = f"/project/pi_miyyer_umass_edu/yekyung/CS646/CS646_Final_Project/data/Qwen_Outputs/nk_{args.retriever}_step_{args.steps}_queries_{args.queries}_docs_{args.docs}_answer.jsonl"
nk_file = f"/project/pi_miyyer_umass_edu/yekyung/CS646/CS646_Final_Project/data/Qwen_Outputs/nk_{args.retriever}_step_{args.steps}_queries_{args.queries}_docs_{args.docs}_output.jsonl"
with open(nk_file, "r") as f:
    nk_data = [json.loads(x.strip()) for x in f.readlines() if x.strip()]
print("Loaded nk_data.")

"""
GENERATE RESPONSE
"""
qwen_model = QwenGeneration()
print("Qwen model loaded.")
with open(nk_answer_file, "w") as f:
    for dict_item in tqdm(nk_data):
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

        f.write(json.dumps(dict_item) + "\n")
