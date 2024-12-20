import pdb
import json
import argparse

from tqdm import tqdm

from RetrievalEval import RetrievalEval

# add baseline_name to the parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "--file",
    type=str,
    help="path to the file containing the retrieval results"
)
parser.add_argument(
    "--key",
    type=str,
    help="key to the retrieval results in the file"
)
args = parser.parse_args()

"""
LOAD FRAMES DATA WITH RETRIEVED LINKS
"""
print(f"file_path: {args.file}")
print(f"key_to_links: {args.key}")

with open(args.file, "r") as f:
    retrieval_data = [json.loads(x.strip()) for x in f.readlines() if x.strip()]

if 'nk_' in args.file:
    retrieved_results = []
    for dict_item in tqdm(retrieval_data):
        url_score = {}
        retrieval_history = dict_item['retrieval_history']
        for retrieved in retrieval_history:
            retrieved_docs = retrieved['retrieved_docs']
            for docs in retrieved_docs:
                for doc in docs:
                    url = doc[0]
                    score = doc[1]
                    if url in url_score:
                        if score > url_score[url]:
                            url_score[url]=score
                    else:
                        url_score[url] = score

        sorted_data = dict(sorted(url_score.items(), key=lambda item: item[1], reverse=True))
        retrieved_results.append(list(sorted_data.keys()))
else:
    retrieved_results = []
    for item in retrieval_data:
        retrieved_link_score_lst = item[args.key]
        retrieved_links = [x[0] for x in retrieved_link_score_lst]
        retrieved_results.append(retrieved_links)
"""
CALCULATE RETRIEVAL METRICS
"""


ground_truth = [x["wiki_links"] for x in retrieval_data] # list of lists

assert len(retrieved_results) == len(ground_truth), \
    "The counts do not match the total number of retrieved results."

retrieval_metrics = RetrievalEval(retrieved_results, ground_truth, k=5)
recall_at_k = retrieval_metrics.recall_at_k()
precision_at_k = retrieval_metrics.precision_at_k()
f1_at_k = retrieval_metrics.f1_at_k()
map = retrieval_metrics.mean_average_precision()

print(f"  Rec@5:\t{recall_at_k:.4f}")
print(f"  Prec@5:\t{precision_at_k:.4f}")
print(f"  F1@5:\t{f1_at_k:.4f}")
print(f"  MAP:\t{map:.4f}\n")
