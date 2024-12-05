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
    "--file",
    type=str,
    help="path to the file containing the retrieval results"
)
parser.add_argument(
    "--key",
    type=str,
    help="key to the retrieval results in the file"
)
parser.add_argument(
    "--alias",
    type=str,
    help="alias for the output file"
)
args = parser.parse_args()

"""
LOAD FRAMES DATA WITH QWEN ANSWERS
"""
print(f"file_path: {args.file}")
print(f"key_to_ans: {args.key}")

with open(args.file, "r") as f:
    frames_data = [json.loads(x.strip()) for x in f.readlines() if x.strip()]

if "bm25" in args.file.lower():
    retriever = "bm25"
elif "dpr" in args.file.lower():
    retriever = "dpr"
else:
    raise ValueError("Retriever not found in file name.")

"""
GENERATE RESPONSE
"""
gpt = GPTGeneration()

out_dir = "/project/pi_miyyer_umass_edu/yekyung/CS646/CS646_Final_Project/data/Auto_Rater_Outputs"
alias = f"gpt4o_mini_judgment_{args.alias}"
judgment_out_file = os.path.join(out_dir, f"{alias}.jsonl")
print(f"output file: {judgment_out_file}")

start_point = utils.get_start_point(judgment_out_file)

# gpt-4o-mini pricing:
#   $0.150 / 1M input tokens
#   $0.600 / 1M output tokens
with open(judgment_out_file, "a") as f:
    all_id = set()
    for dict_item in tqdm(frames_data[start_point:]):
        query_id = dict_item['ID']
        if query_id in all_id:
            print("duplicated ID")
            continue
        question = dict_item["Prompt"]
        ground_truth = dict_item["Answer"]
        predicted_answer = dict_item[args.key]

        prompt = utils.auto_eval_prompt_template.format(
            question=question,
            ground_truth_answer=ground_truth,
            predicted_answer=predicted_answer
        )

        judgment, _, _ = gpt.get_response(prompt, judge=True)
        
        dict_item[f"gpt4o_mini_{retriever}_judgment"] = judgment
        f.write(json.dumps(dict_item) + "\n")
