"""
This script is to get the no rag responses for the lower bound of the model performance.
"""

import os
import ast
import pdb
import json
import utils
from tqdm import tqdm
from QwenGeneration import QwenGeneration

"""
LOAD DATA
"""
frames_file = "data/Qwen_Outputs/oracle_output.jsonl"
with open(frames_file, "r") as f:
    frames_data = [json.loads(x.strip()) for x in f.readlines() if x.strip()]
print("Loaded frames_data.")

"""
GENERATE RESPONSE
"""
qwen_model = QwenGeneration()
print("Qwen model loaded.")

oracle_output_file = "data/Qwen_Outputs/zero_shot_and_oracle_output.jsonl"
start_point = utils.get_start_point(oracle_output_file)

with open(oracle_output_file, "a") as f:
    for dict_item in tqdm(frames_data[start_point:]):
        query = dict_item["Prompt"]

        response = qwen_model.get_response(query)
        grouth_truth_ans = dict_item["Answer"]

        print(f"Ground truth answer: {grouth_truth_ans}")
        print(f"Qwen response: {response}")

        dict_item["Qwen_zero_shot_answer"] = response.strip()

        f.write(json.dumps(dict_item) + "\n")
