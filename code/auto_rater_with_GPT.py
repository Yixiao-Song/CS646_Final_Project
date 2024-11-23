import os
import pdb
import json
import utils
import argparse
from tqdm import tqdm
from GPTGeneration import GPTGeneration

# add baseline_name to the parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "--type",
    type=str,
    default="oracle",
    help="dict key to get the answer (oracle, zero_shot, bm25, dpr)"
    )
args = parser.parse_args()

"""
LOAD FRAMES DATA WITH QWEN ANSWERS
"""
if args.type == "zero_shot":
    file_path = "data/Qwen_Outputs/zero_shot_and_oracle_output.jsonl"
    key_to_ans = "Qwen_zero_shot_answer"
elif args.type == "oracle":
    file_path = "data/Qwen_Outputs/zero_shot_and_oracle_output.jsonl"
    key_to_ans = "Qwen_oracle_answer"
elif args.type == "bm25":
    file_path = "data/Qwen_Outputs/naive_BM25_output.jsonl"
    key_to_ans = "Qwen_naive_BM25_answer"
elif args.type == "dpr":
    file_path = "data/Qwen_Outputs/naive_DPR_output.jsonl"
    key_to_ans = "Qwen_naive_DPR_answer"

print(f"file_path: {file_path}")
print(f"key_to_ans: {key_to_ans}")

with open(file_path, "r") as f:
    frames_data = [json.loads(x.strip()) for x in f.readlines() if x.strip()]

"""
GENERATE RESPONSE
"""
gpt = GPTGeneration()

judgment_out_file = f"data/Auto_Rater_Outputs/gpt4o_mini_judgment_{args.type}.jsonl"
start_point = utils.get_start_point(judgment_out_file)

# gpt-4o-mini pricing:
#   $0.150 / 1M input tokens
#   $0.600 / 1M output tokens
with open(judgment_out_file, "a") as f:
    for dict_item in tqdm(frames_data[start_point:]):
        question = dict_item["Prompt"]
        ground_truth = dict_item["Answer"]
        predicted_answer = dict_item[key_to_ans]

        prompt = utils.auto_eval_prompt_template.format(
            question=question,
            ground_truth_answer=ground_truth,
            predicted_answer=predicted_answer
        )

        judgment, _, _ = gpt.get_response(prompt)
        
        dict_item[f"gpt4o_mini_{args.type}_judgment"] = judgment
        f.write(json.dumps(dict_item) + "\n")
