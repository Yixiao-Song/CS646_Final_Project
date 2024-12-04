#!/bin/bash -l

module load uri/main
module load Java/21.0.2
source /project/pi_miyyer_umass_edu/yixiao/CS646/FinalProject/final_proj_env/bin/activate

cd /work/pi_miyyer_umass_edu/yapeichang/CS646_Final_Project

python3 code/retrieval_eval.py \
    --file data/Qwen_Outputs/decomp_bm25_answers_maxq3_topkNone_ndocs5.jsonl \
    --key Qwen_decomp_bm25_direct_links

python3 code/retrieval_eval.py \
    --file data/Qwen_Outputs/decomp_bm25_answers_maxq5_topk1_ndocs5.jsonl \
    --key Qwen_decomp_bm25_direct_links \
    --eval_type ndcg
