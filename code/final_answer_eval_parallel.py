import os
import pdb
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    “--file”,
    type=str
)
parser.add_argument(
    “--key”,
    type=str,
    default=“gpt4o_mini_bm25_judgment”
)
args = parser.parse_args()

with open(args.file, “r”) as f:
    judgment_data = [json.loads(x.strip()) for x in f.readlines() if x.strip()]

yes_cnt = 0
no_cnt = 0
for dict_item in judgment_data:
    if dict_item[“Qwen_decomp_bm25_direct_answer”] == “SKIPPED”:
        continue
    judgment = dict_item[args.key]
    if “yes” in judgment.lower():
        yes_cnt += 1
    elif “no” in judgment.lower():
        no_cnt += 1

total = yes_cnt + no_cnt

print(f”Total: {total}, Yes: {yes_cnt}, No: {no_cnt}“)
accuracy = yes_cnt / len(judgment_data)
print(f”Accuracy: {100*accuracy:.2f}%“)
