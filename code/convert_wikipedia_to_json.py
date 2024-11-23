"""
This script is written to convert the data to json. 
dictionary 1:
    key = url
    val = contents
dictionary 2:
    key = first 200 characters of contents
    val = url
"""

import json
from tqdm import tqdm

wiki_jsonl_file = "data/wikipedia/jsonl_output/wikipedia_filtered.jsonl"
with open(wiki_jsonl_file, "r") as f:
    wiki_data = [json.loads(x.strip()) for x in f.readlines() if x.strip()]

wiki_url_content_dict = {}
wiki_content_url_dict = {}
for dict_item in tqdm(wiki_data):
    url = dict_item["url"]
    key_dict = {
        "id": dict_item["id"],
        "title": dict_item["title"],
        "contents": dict_item["contents"]
    }

    wiki_url_content_dict[url] = key_dict
    wiki_content_url_dict[dict_item["contents"][:200]] = url

wiki_url_content_out_file = "data/wikipedia/jsonl_output/wikipedia_filtered_url_to_content.json"
with open(wiki_url_content_out_file, "w") as f:
    json.dump(wiki_url_content_dict, f)

wiki_content_url_file = "data/wikipedia/jsonl_output/wikipedia_filtered_content_to_url.json"
with open(wiki_content_url_file, "w") as f:
    json.dump(wiki_content_url_dict, f)
