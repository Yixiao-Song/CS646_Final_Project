import pdb
import json
import utils
from tqdm import tqdm
from urllib.parse import unquote, quote


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
GET FILTERED WIKIPEDIA LINKS
"""
wiki_filtered_file = "data/wikipedia/jsonl_output/wikipedia_filtered.jsonl"
with open(wiki_filtered_file, "r") as f:
    wiki_data = [json.loads(x.strip()) for x in f.readlines() if x.strip()]

wiki_links = []
for dict_item in wiki_data:
    wiki_links.append(dict_item['url'])

print(f"There are {len(wiki_links)} filtered wiki articles.")

"""
FIND UNMATCHED URLs
"""
not_in_frames_cnt = 0
for link in tqdm(wiki_links):
    try:
        frames_wiki_links.remove(link)
    except:
        not_in_frames_cnt += 1
        continue

print(f"not_in_frames_cnt: {not_in_frames_cnt}")
print(f"len(frames_wiki_links) unmatched: {len(frames_wiki_links)}")

"""
FILTER OUT FRAMES DATA THAT CONTAIN THE UNMATCHED URLS
"""

filtered_frames_data = []
for dict_item in frames_data:
    wiki_links_lst = dict_item['wiki_links']
    if any(link in frames_wiki_links for link in wiki_links_lst):
        continue
    filtered_frames_data.append(dict_item)

print(f"There are {len(filtered_frames_data)} filtered frames data points.")

"""
GET THE FILTERED FRAMES LINKS
"""
filtered_frames_wiki_links = []
for dict_item in filtered_frames_data:
    wiki_links_lst = dict_item['wiki_links']
    filtered_frames_wiki_links.extend(wiki_links_lst)

print(f"There are {len(filtered_frames_wiki_links)} filtered wiki links "
        f"in {len(filtered_frames_data)} data points.")

"""
SAVE THE FILTERED FRAMES DATA
"""
filtered_frames_file = "data/frames_dataset_2_5_links_filtered.jsonl"
with open(filtered_frames_file, "w") as f:
    for dict_item in filtered_frames_data:
        f.write(json.dumps(dict_item, ensure_ascii=False) + '\n')

"""
CHECK FILTERED FRAMES DATA'S URLS ARE ALL IN FILTERED WIKI DATA
"""
for link in filtered_frames_wiki_links:
    if link not in wiki_links:
        print(link)
        pdb.set_trace()
