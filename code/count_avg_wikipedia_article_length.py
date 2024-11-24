import pdb
import json
import numpy as np
from GPTGeneration import GPTGeneration

wikipedia_file = "data/wikipedia/jsonl_output/wikipedia_filtered_url_to_content.json"
with open(wikipedia_file, "r") as f:
    wiki_dictionary = json.load(f)

content_lst = [x["contents"] for x in list(wiki_dictionary.values())]

gpt = GPTGeneration()
tok_len_lst = []
for content_item in content_lst:
    tok_len_lst.append(gpt.tok_count(content_item))

mean_tok_len = np.mean(tok_len_lst)
median_tok_len = np.median(tok_len_lst)
max_tok_len = np.max(tok_len_lst)
min_tok_len = np.min(tok_len_lst)
print(f"Mean token length: {mean_tok_len:.2f}")
print(f"Median token length: {median_tok_len}")
print(f"Max token length: {max_tok_len}")
print(f"Min token length: {min_tok_len}")
