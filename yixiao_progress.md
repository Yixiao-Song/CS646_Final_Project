1. Downloaded FRAMES from HuggingFace, turned tsv to jsonl. Filtered to queries with 2-5 wiki links.
    - code: `code/convert_tsv_to_jsonl.py`
    - data: `data/frames_dataset_2_5_links.jsonl` (769 dps)

2. Download the en Wikipedia split from [HuggingFace](https://huggingface.co/datasets/graelo/wikipedia)
    - code: `code/convert_wikipedia_to_json.py`
    - data: `data/wikipedia/graelo___wikipedia`

3. Convert Wikipedia from binary to jsonl for indexing. Jsonl only contains the wikipedia article used in frames and 795 random articles.
    - code: `code/convert_wikipedia_to_jsonl.py`
    - data: `data/wikipedia/jsonl_output/wikipedia_filtered.jsonl`
    - Notes: 
        1. The urls in frames are readable urls (with some umlauts using special characters) but the ones in wikipedia use special characters for things like underscore and umlauts. To unify their format, I normalized the urls.
        2. Some urls in frames contain `#` which points to a specific section in a wikipedia article. When filtering links, the frames urls are split and only the first part (i.e., the page link) is kept.
        3. Some urls are in the mobile format (`en.m.wikipedia` instead of `en.wikipedia`). These are converted from mobile to web format.
        4. No matter how hard I tried, there are **200** frames wiki link that cannot be matched in the wikipedia data although these frames wiki links indeed exit. One reason for that is the following: The paper said it uses the [20230601en dump](https://tinyurl.com/36jxum2y) but some links (e.g., https://en.wikipedia.org/wiki/Tyson_Fury_vs_Oleksandr_Usyk) are about events happened in 2024.

4. Filter FRAMES based on the unmatched urls
    - code: `code/remove_data_with_unmatched_urls_from_frames.py`
    - data: `data/frames_dataset_2_5_links_filtered.jsonl` (528 dps)

**Interm Summary**:
- FRAMES data points: 528
- Wikipedia articles in the jsonl file: 2005
- Wikipedia jsonl: `data/wikipedia/jsonl_output/wikipedia_filtered.jsonl`
    Note: I did not go back to further filter wikipedia to contain only the 1496 articles used in the 528 data points. 
- FRAMES jsonl: `data/frames_dataset_2_5_links_filtered.jsonl`

4. Index Wikipedia
    - code: 
        python -m pyserini.index.lucene \
        --collection JsonCollection \
        --input data/wikipedia/jsonl_output \
        --index data/wikipedia/index_output \
        --generator DefaultLuceneDocumentGenerator \
        --threads 4 \
        --storePositions --storeDocvectors --storeRaw
    - data: `data/wikipedia/index_output`

5. Set up Qwen2.5-14B-Instruct generation class
    - code: `code/GetResponseQwen14B.py`

6. Get Oracle results
    - code: `code/oracle.py`



draft: 
5. Set up Qwen generation script
[x] pip install vllm
- export HF_HOME=/scratch/workspace/mkarpinska_umass_edu-nocha_llms
-   import argparse
    import os
    import pickle
    import logging
    from tqdm import tqdm
    import json
    import torch
    # from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    import time
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    import ray
    import csv
    import sys
- Load model code: 
  - `model_id` is just hf id (e.g., `--model_id Qwen/Qwen2.5-1.5B-Instruct`)
  - `tensor_parallel_size` is the num of gpus (no odd num, it’s either nothing there (for 1 gpu) do 2 or 4 or 6)
  - `trust_remote_code=True` may need for some models, can omit and see if it complains
    def load_model_and_tokenizer(model_id):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        ray.shutdown()
        ray.init(num_gpus=torch.cuda.device_count())

        if '72' in model_id or '70' in model_id:
            llm = LLM(model=model_id, tensor_parallel_size=4, trust_remote_code=True)
        elif '14' in model_id or "medium" in model_id or '32' in model_id or '512' in model_id:
            llm = LLM(model=model_id, tensor_parallel_size=2, trust_remote_code=True)
        else:
            llm = LLM(model=model_id, trust_remote_code=True)

        return llm, tokenizer

- for the model_call, pass the llm and tokenizer returned
  (set hf path to the folder containing the models `export HF_HOME=/scratch/workspace/mkarpinska_umass_edu-nocha_llms`)

    def model_call(prompt, llm, tokenizer, ctx_window=128000, temp=0.0, max_tokens=800):
        sampling_params = SamplingParams(temperature=temp, max_tokens=max_tokens)

        messages = [
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        outputs = llm.generate([text], sampling_params)

        for output in outputs:
            generated_text = output.outputs[0].text
            print(f"Generated text: {generated_text!r}", flush=True)

        return generated_text
- To make sure vllm sees the correct num of gpus
  `ray.shutdown()`
  `ray.init(num_gpus=torch.cuda.device_count())`
- module load cuda/11.8
- Get oracle resultscd