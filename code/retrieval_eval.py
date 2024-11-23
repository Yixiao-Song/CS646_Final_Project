import pdb
import json
import argparse
from RetrievalEval import RetrievalEval

# add baseline_name to the parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "--type",
    type=str,
    default="bm25",
    help="dict key to get the retrieved links (bm25 or dpr)"
    )
args = parser.parse_args()

"""
LOAD FRAMES DATA WITH RETRIEVED LINKS
"""
if args.type == "bm25":
    file_path = "data/Qwen_Outputs/naive_rag_baseline_BM25_retrieve.jsonl"
    key_to_links = "naive_rag_retrieve_results_BM25"
elif args.type == "dpr":
    file_path = "data/Qwen_Outputs/naive_rag_baseline_DPR_retrieve.jsonl"
    key_to_links = "naive_rag_retrieve_results_DPR"

print(f"file_path: {file_path}")
print(f"key_to_links: {key_to_links}")

with open(file_path, "r") as f:
    retrieval_data = [json.loads(x.strip()) for x in f.readlines() if x.strip()]

"""
CALCULATE RETRIEVAL METRICS
"""
retrieved_results = []
for item in retrieval_data:
    retrieved_link_score_lst = item[key_to_links]
    retrieved_links = [x[0] for x in retrieved_link_score_lst]
    retrieved_results.append(retrieved_links)

ground_truth = [x["wiki_links"] for x in retrieval_data] # list of lists

assert len(retrieved_results) == len(ground_truth), \
    "The counts do not match the total number of retrieved results."

retrieval_metrics = RetrievalEval(retrieved_results, ground_truth, k=5)
recall_at_k = retrieval_metrics.recall_at_k()
precision_at_k = retrieval_metrics.precision_at_k()
f1_at_k = retrieval_metrics.f1_at_k()
map = retrieval_metrics.mean_average_precision()

print(f"Rec@5:\t{recall_at_k:.4f}")
print(f"Prec@5:\t{precision_at_k:.4f}")
print(f"F1@5:\t{f1_at_k:.4f}")
print(f"MAP:\t{map:.4f}")
