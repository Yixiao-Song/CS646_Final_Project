"""
This script is written to do the following:
1) download the en split of the Wikipedia data
2) convert the data to jsonl for indexing

The wikipedia data is filtered to contain only the wikipedia needed for frames
and 795 random articles.
"""

import os
import pdb
import json
import utils
import random
from tqdm import tqdm
from datasets import load_dataset


# def normalize_url(url):
#     """Normalize URLs by decoding and re-encoding them

#     Args:
#         url: the unnormalized url from wikipedia dataset

#     Return:
#         normalized_url: readable url link
#     """
#     # Decode percent-encoded parts
#     decoded_url = unquote(url)
#     # Re-encode spaces as underscores (for Wikipedia consistency)
#     normalized_url = decoded_url.replace(" ", "_")
#     return normalized_url


"""
DOWNLOAD WIKIPEDIA
"""
cache_dir = "data/wikipedia"
wikipedia_en = load_dataset(
    "graelo/wikipedia",
    "20230601.en",
    cache_dir=cache_dir
    )

train_data = wikipedia_en["train"]

# print some basic dataset info
print(f"*** Dataset info ***\n{train_data}")
print(f"*** Dataset keys ***\n{train_data.column_names}\n")

"""
GET WIKIPEDIA LINKS FROM FRAMES
"""
frames_file = "/project/pi_miyyer_umass_edu/yixiao/CS646/FinalProject/data/frames_dataset_2_5_links.jsonl"
with open(frames_file, "r") as f:
    frames_data = [json.loads(x.strip()) for x in f.readlines() if x.strip()]

frames_wiki_links = []
for dict_item in frames_data:
    wiki_links_lst = dict_item['wiki_links']

    frames_wiki_links.extend(wiki_links_lst)

print(f"There are {len(frames_wiki_links)} wiki links "
      f"in {len(frames_data)} data points.")

"""
CONVERT WIKIPEDIA TO JSONL
"""

jsonl_out_file = "data/wikipedia/jsonl_output/wikipedia_filtered.jsonl"

found_links_cnt = 0
in_frames = 0
with open(jsonl_out_file, "w") as f:
    for i, doc in tqdm(enumerate(train_data)):
        # filter criteria
        url = doc.get("url", "")
        
        if url == "":
            print("--> No url.")
            continue
        
        normalized_url = utils.normalize_url(url)
        if normalized_url not in frames_wiki_links:
            continue

        # construct jsonl line
        jsonl_line = {
            "id": doc["id"],
            "url": normalized_url,
            "title": doc["title"],
            "contents": doc["text"]  
        }

        found_links_cnt += 1
        print(f"found_links_cnt: {found_links_cnt}")

        # write the jsonl line
        json.dump(jsonl_line, f)
        f.write("\n")

print(f"Filtered and stored {found_links_cnt} wikipedia articles.")
