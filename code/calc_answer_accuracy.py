import os
import pdb
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--type",
    type=str,
    default="oracle",
    help="dict key to get the answer (oracle, zero_shot, bm25, dpr)"
    )
parser.add_argument(
    "--alias",
    type=str,
    help="alias for the output file"
)
args = parser.parse_args()

file_path = f"/project/pi_miyyer_umass_edu/yekyung/CS646/CS646_Final_Project/data/Auto_Rater_Outputs/gpt4o_mini_judgment_{args.alias}.jsonl"
with open(file_path, "r") as f:
    judgment_data = [json.loads(x.strip()) for x in f.readlines() if x.strip()]

yes_cnt = 0
no_cnt = 0
for dict_item in judgment_data:
    judgment = dict_item[f"gpt4o_mini_{args.type}_judgment"]

    if "yes" in judgment.lower():
        yes_cnt += 1
    elif "no" in judgment.lower():
        no_cnt += 1
    else:
        print(judgment.lower())

assert yes_cnt + no_cnt == len(judgment_data), \
    "The counts do not match the total number of judgments."

accuracy = yes_cnt / len(judgment_data)
print(f"Type: {args.type}, Accuracy: {100*accuracy:.2f}%")
