# From Complexity to Clarity: Enhancing Factuality, Retrieval, and Reasoning through Sub-Query Decomposition

## Environment setup

All experiments were run on Unity. To set up the environment, run:

```
module load uri/main
module load Java/21.0.2
source /project/pi_miyyer_umass_edu/yixiao/CS646/FinalProject/final_proj_env/bin/activate
```

Alternatively, you can build your own venv using `requirements.txt`.

## Running baselines and evaluations

Please refer to [this file](https://github.com/Yixiao-Song/CS646_Final_Project/blob/master/yixiao_progress.md) for details on running baseline experiments and evaluations.

## Running parallel retrieval

### Query decomposition

Run zero-shot fixed-`n` decomposition:

```
python3 code/query_extraction.py --max_num n
```

Run few-shot dynamic-`n` decomposition:

```
python3 code/query_extraction_fewshot.py
```

### Retrieval and aggregation

Run a `*n-k-m` configuration:

```
python3 code/decomp.py --maxq n --topk k --ndocs m
```

This code will generate all final contexts and final answers. If you want to use the queries decomposed with the few-shot method, unset `--maxq` (it defaults to `None`).

## Running incremental retrieval

To run incremental retrieval and get final contexts, run:

```
python3 code/nk.py --steps s --queries n --docs k
```

To get the final answers with Qwen-2.5-14B-Instruct, run:

```
python3 code/nk_generate_final_answer.py --steps s --queries n --docs k
```
