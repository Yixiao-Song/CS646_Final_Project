import os
import ast
import pdb
import json
import utils
import argparse
from tqdm import tqdm
from GPTGeneration import GPTGeneration

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_num",
        default="5",
        help="Max number of extracted sub-query"
    )
    args = parser.parse_args()

    project_path = '/project/pi_miyyer_umass_edu/yekyung/CS646/CS646_Final_Project'
    frames_file = f"{project_path}/data/frames_dataset_2_5_links_filtered_subquery.jsonl"
    with open(frames_file, "r") as f:
        frames_data = [json.loads(x.strip()) for x in f.readlines() if x.strip()]
    print("Loaded frames_data")

    max_num = args.max_num

    base_path = f"{project_path}/data/"
    generate_output = f"{base_path}/frames_dataset_2_5_links_filtered_extracted_queries_max_{max_num}.jsonl"


    """
    GENERATE RESPONSE
    """
    gpt_model = GPTGeneration(max_tokens=512)
    print("GPT4 model loaded.")
    print(f"Starting decompose query into max {max_num} queries")

    with open(generate_output, "w") as f:
        for dict_item in tqdm(frames_data):
            query = dict_item["Prompt"]

            prompt = utils.extract_query_prompt_template.format(
                max_num = max_num,
                question=query
            )

            response, _,_ = gpt_model.get_response(prompt)
            sub_query = json.loads(response)['atomic_questions']
            dict_item["extracted_subquery"] = sub_query
            f.write(json.dumps(dict_item) + "\n")

if __name__ == "__main__":
    main()