import os
import ast
import pdb
import json
import utils
from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher

"""
LOAD DATA
"""
frames_file = "data/frames_dataset_2_5_links_filtered.jsonl"
with open(frames_file, "r") as f:
    frames_data = [json.loads(x.strip()) for x in f.readlines() if x.strip()]
print("Loaded frames_data.")

"""
SET UP BM25 Retriever
"""
searcher = LuceneSearcher('data/wikipedia/index_output_filtered_wiki')
def BM25_searcher(query):
    hits = searcher.search(query, k=5)
    return hits

out_file = "data/naive_rag_baseline_BM25_retrieve.jsonl"
with open(out_file, "w") as f:
    for dict_item in tqdm(frames_data):
        query = dict_item["Prompt"]
        print(f"query: {query}")
        hits = BM25_searcher(query)

        url_score_tuple_lst = []
        for i in range(len(hits)):
            docid = hits[i].docid
            score = hits[i].score

            raw_doc_dict = ast.literal_eval(searcher.doc(docid).raw())

            url_score_tuple_lst.append((raw_doc_dict['url'], score))

        dict_item["naive_rag_retrieve_results_BM25"] = url_score_tuple_lst
        f.write(json.dumps(dict_item) + "\n")
