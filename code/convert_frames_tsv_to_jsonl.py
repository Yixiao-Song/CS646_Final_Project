"""
This script is written to convert the tsv file of the frames data to a jsonl
file for easy access later.
"""

import os
import pdb
import csv
import json

input_tsv = "data/frames_dataset.tsv"
output_jsonl = input_tsv.replace(".tsv", ".jsonl")

with open(input_tsv, 'r', encoding='utf-8') as tsv_file, \
    open(output_jsonl, 'w', encoding='utf-8') as jsonl_file:
    reader = csv.DictReader(tsv_file, delimiter='\t')
    
    for row in reader:
        jsonl_file.write(json.dumps(row) + '\n')

print(f"TSV data from {input_tsv} has been converted to JSONL and saved in "
      f"{output_jsonl}.")
