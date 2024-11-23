import pdb
import json

frames_file = "data/Auto_Rater_Outputs/gpt4o_mini_judgment_oracle.jsonl"
with open(frames_file, "r") as f:
    frames_data = [json.loads(x.strip()) for x in f.readlines() if x.strip()]

all_lines_lst = []
for dict_item in frames_data:
    question = dict_item["Prompt"].replace("\n", " ")
    ground_truth = dict_item["Answer"].replace("\n", " ")
    oracle_answer = dict_item["Qwen_oracle_answer"].replace("\n", " ")
    judgment = dict_item["gpt4o_mini_oracle_judgment"].replace("\n", " ")

    line_lst = [question, ground_truth, oracle_answer, judgment]
    all_lines_lst.append(line_lst)

out_file = "data/Auto_Rater_Outputs/evaluate_gpt4o_mini_judgment_oracle.tsv"
with open(out_file, "w") as f:
    f.write("Question\tGround Truth\tOracle Answer\tJudgment\n")
    for line_lst in all_lines_lst:
        f.write("\t".join(line_lst) + "\n")
