"""
This script is written to convert the tsv file of the frames data to a jsonl
file for easy access later.
"""

import os
import ast
import pdb
import csv
import json
import utils
from urllib.parse import unquote, quote


input_tsv = "data/frames_dataset.tsv"
output_jsonl = input_tsv.replace(".tsv", "_2_5_links.jsonl")

out_cnt = 0
with open(input_tsv, 'r', encoding='utf-8') as tsv_file, \
    open(output_jsonl, 'w', encoding='utf-8') as jsonl_file:
    reader = csv.DictReader(tsv_file, delimiter='\t')
    
    for row in reader:
        wiki_links_lst = ast.literal_eval(row['wiki_links'])
        if len(wiki_links_lst) > 5:
            continue
        row['wiki_links'] = [utils.normalize_url(x) for x in wiki_links_lst]
        jsonl_file.write(json.dumps(row, ensure_ascii=False) + '\n')
        out_cnt += 1

print(f"TSV data from {input_tsv} has been converted to JSONL and saved in "
      f"{output_jsonl}.")
print(f"There are {out_cnt} data points in the output.")