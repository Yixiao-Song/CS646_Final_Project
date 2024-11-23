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

5. Index Wikipedia (done on Arkham)
    - code: 
        python -m pyserini.index.lucene \
            --collection JsonCollection \
            --input data/wikipedia/jsonl_output \
            --index data/wikipedia/index_output_filtered_wiki \
            --generator DefaultLuceneDocumentGenerator \
            --threads 4 \
            --storePositions --storeDocvectors --storeRaw
    - data: `data/wikipedia/index_output_filtered_wiki`

**Interm Summary**:
- FRAMES data points: 528
- Wikipedia articles in the jsonl file: 2005
- Wikipedia jsonl: `data/wikipedia/jsonl_output/wikipedia_filtered.jsonl`
    *Note*: I did not go back to further filter wikipedia to contain only the 1496 articles used in the 528 data points. 
- Wikipedia index: `data/wikipedia/index_output_filtered_wiki`
- FRAMES jsonl: `data/frames_dataset_2_5_links_filtered.jsonl`
- It is confirmed that all frames urls are in wikipedia jsonl.

5. Set up Qwen2.5-14B-Instruct generation class
    - code: `code/GetResponseQwen14B.py`
    - hyperparameters: default
    - maximum length: 32768
    - **Important**: set `HF_HOME` in your `~/.bashrc`

6. Get Oracle results (upper bound)
    1. Get a url to contents mapping for getting contents using the FRAMES grouth truth urls
        - code: `code/convert_wikipedia_to_json.py`
        - data: `data/wikipedia/jsonl_output/wikipedia_filtered_url_to_content.json`

    2. Get the oracle results
        - code: `code/oracle.py`
        - data: `data/Qwen_Outputs/oracle_output.jsonl`

7. Get zero-shot results (lower bound)
    - code: `code/zero_shot_baseline.py`
    - data: `data/Qwen_Outputs/zero_shot_and_oracle_output.jsonl`
    - Note: oracle and zero-shot results are combined in the same file.

8. Get naive rag results (BM25)
    - Retrieve
        - code: `code/naive_rag_baseline_BM25_retrieve.py` (done on Arkham)
        - data: `data/Qwen_Outputs/naive_rag_baseline_BM25_retrieve.jsonl`
    - Generate
        - code: `code/naive_rag_baseline_BM25_generate.py`
        - data: `data/Qwen_Outputs/naive_BM25_output.jsonl`
    - Note: naive rag results are in a separate file from the zero-shot and oracle results.
    - Interesting outputs:
        - `However, it's important to note that the phrasing of the question doesn't align correctly with the information given about Dismal Euphony and Queen.`
        - `The context does not mention anything about an artist who released the album "Father of Asahd" attending the same high school as Mark Ruiz, but it is irrelevant to answering the specific question about Mark Ruiz's Olympic participation.`
        - `The question does not provide sufficient information to accurately identify the specific 2002 science fiction novel or the author in question. However, given the mention of La Llorona and themes of personal identity, and considering Philip K. Dick's influence and works that incorporate similar themes, it is possible the author being referred to could be someone influenced by or working in a similar thematic space.\nPhilip K. Dick wrote a trilogy under the publisher Doubleday (before switching to Ballantine Books for much of his career). The trilogy in question is the "Three Stigmata of Palmer Eldritch"/"Ubik"/"Flow My Tears, the Policeman Said" series. However, these were written in the 1960s and 1970s, not in 2002.\nAnother possibility is James Patrick Kelly, who wrote a novel called "Burn" in 2002 which could potentially fit the description, but he did not write a trilogy under the same publisher that year.\nGiven the constraints and the lack of precise details, the most accurate response would be that the specific author and trilogy cannot be definitively identified from the given information. The author and trilogy in question likely wrote under Tor Books, a common publisher for science fiction novels referencing La Llorona and exploring themes of personal identity in the early 2000s. However, without more specific details, it's impossible to name the exact trilogy.`

9. Get naive rag results (DPR)
    - Retrieve
        - code: 

    - Note: DPR can only encode 512 tokens, which is way smaller than most of the documents in wikipedia.

10. Retrieval metrics

9. Auto-rater
    - code: 
    - data: 
    - Note: I tried with Qwen2.5 7B and 14B models as the auto-rater using the prompt in `utils.py` (`auto_eval_prompt_template`). But the model does not follow the instruction at all. Simple heuristic such as `ground_truth.lower() in qwen_answer.lower()` also does not work because some ground truth answers are sentences.

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