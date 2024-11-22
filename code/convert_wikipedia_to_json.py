"""
This script is written to convert the data to json. 
key = url
val = contents
"""

# pip install datasets

import os
import pdb
import json
from tqdm import tqdm

wiki_jsonl_file = "data/wikipedia/jsonl_output/wikipedia_filtered.jsonl"
with open(wiki_jsonl_file, "r") as f:
    wiki_data = [json.loads(x.strip()) for x in f.readlines() if x.strip()]

wiki_url_content_dict = {}
for dict_item in tqdm(wiki_data):
    url = dict_item["url"]
    key_dict = {
        "id": dict_item["id"],
        "title": dict_item["title"],
        "contents": dict_item["contents"]
    }

    wiki_url_content_dict[url] = key_dict

wiki_url_content_out_file = "data/wikipedia/jsonl_output/wikipedia_filtered_url_to_content.json"
with open(wiki_url_content_out_file, "w") as f:
    json.dump(wiki_url_content_dict, f)
