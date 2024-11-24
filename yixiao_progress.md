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

4. Construct two wikipedia dictionaries (saved in 2 json files): (1) mapping `url` to `contents`, (2) mapping stripped first 200 `contents` characters to `url`.
    - code: `code/convert_wikipedia_to_json.py`
    - data: `data/wikipedia/jsonl_output/wikipedia_filtered_content_to_url.json`, `data/wikipedia/jsonl_output/wikipedia_filtered_url_to_content.json`

5. Filter FRAMES based on the unmatched urls
    - code: `code/remove_data_with_unmatched_urls_from_frames.py`
    - data: `data/frames_dataset_2_5_links_filtered.jsonl` (528 dps)

6. Index Wikipedia (done on Arkham)
    - code: 
        python -m pyserini.index.lucene \
            --collection JsonCollection \
            --input data/wikipedia/jsonl_output \
            --index data/wikipedia/index_output_filtered_wiki \
            --generator DefaultLuceneDocumentGenerator \
            --threads 4 \
            --storePositions --storeDocvectors --storeRaw
    - data: `data/wikipedia/index_output_filtered_wiki`

7. **Interm Summary**:
    - FRAMES data points: 528
    - Wikipedia articles in the jsonl file: 2005
    - Wikipedia jsonl: `data/wikipedia/jsonl_output/wikipedia_filtered.jsonl`
        *Note*: I did not go back to further filter wikipedia to contain only the 1496 articles used in the 528 data points. 
    - Wikipedia index: `data/wikipedia/index_output_filtered_wiki`
    - FRAMES jsonl: `data/frames_dataset_2_5_links_filtered.jsonl`
    - It is confirmed that all frames urls are in wikipedia jsonl.

8. Set up Qwen2.5-14B-Instruct generation class
    - code: `code/GetResponseQwen14B.py`
    - hyperparameters: default
    - maximum length: 32768
    - **Important**: set `HF_HOME` in your `~/.bashrc`

9. Get Oracle results (upper bound)
    1. Get a url to contents mapping for getting contents using the FRAMES grouth truth urls
        - code: `code/convert_wikipedia_to_json.py`
        - data: `data/wikipedia/jsonl_output/wikipedia_filtered_url_to_content.json`

    2. Get the oracle results
        - code: `code/oracle.py`
        - data: `data/Qwen_Outputs/oracle_output.jsonl`

10. Get zero-shot results (lower bound)
    - code: `code/zero_shot_baseline.py`
    - data: `data/Qwen_Outputs/zero_shot_and_oracle_output.jsonl`
    - Note: oracle and zero-shot results are combined in the same file.

11. Get naive rag results (BM25)
    - Retrieve
        - code: `code/naive_rag_baseline_BM25_retrieve.py` (done on Arkham)
        - data: `data/Qwen_Outputs/naive_rag_baseline_BM25_retrieve.jsonl`
    - Generate
        - code: `code/naive_rag_baseline_BM25_generate.py`
        - data: `data/Qwen_Outputs/naive_BM25_output.jsonl`
    - Note: naive rag results are in a separate file from the zero-shot and oracle results. The top 5 documents are retrieved and used as the context for generating answers.
    - Interesting outputs:
        - `However, it's important to note that the phrasing of the question doesn't align correctly with the information given about Dismal Euphony and Queen.`
        - `The context does not mention anything about an artist who released the album "Father of Asahd" attending the same high school as Mark Ruiz, but it is irrelevant to answering the specific question about Mark Ruiz's Olympic participation.`
        - `The question does not provide sufficient information to accurately identify the specific 2002 science fiction novel or the author in question. However, given the mention of La Llorona and themes of personal identity, and considering Philip K. Dick's influence and works that incorporate similar themes, it is possible the author being referred to could be someone influenced by or working in a similar thematic space.\nPhilip K. Dick wrote a trilogy under the publisher Doubleday (before switching to Ballantine Books for much of his career). The trilogy in question is the "Three Stigmata of Palmer Eldritch"/"Ubik"/"Flow My Tears, the Policeman Said" series. However, these were written in the 1960s and 1970s, not in 2002.\nAnother possibility is James Patrick Kelly, who wrote a novel called "Burn" in 2002 which could potentially fit the description, but he did not write a trilogy under the same publisher that year.\nGiven the constraints and the lack of precise details, the most accurate response would be that the specific author and trilogy cannot be definitively identified from the given information. The author and trilogy in question likely wrote under Tor Books, a common publisher for science fiction novels referencing La Llorona and exploring themes of personal identity in the early 2000s. However, without more specific details, it's impossible to name the exact trilogy.`

12. Get naive rag results (DPR)
    - Retrieve
        - code: `code/naive_rag_baseline_DPR_retrieve.py`
        - saved embeddings and index: `data/embeddings`
        - data: `data/Qwen_Outputs/naive_rag_baseline_DPR_retrieve.jsonl`
    - Note: DPR can only encode 512 tokens, which is way smaller than most of the documents in wikipedia.

    - Generate
        - code: `code/naive_rag_baseline_DPR_generate.py`
        - data: `data/Qwen_Outputs/naive_DPR_output.jsonl`
    - Note: Same as BM25, the top 5 documents are retrieved and used as the context for generating answers.
    - Interesting outputs:
        - `The question seems to be asking for the population of the birthplace of the writer of "Culdcept Saga," but the provided context does not contain any information about the writer of "Culdcept Saga" or their birthplace. Therefore, I cannot accurately provide the population of their birthplace based on the given context. If you could provide additional details or clarify the request, I would be happy to assist further. However, based on the instruction format, I'll assume a generic response since specific data isn't available:\n1 million`
        - `The given context does not contain information about the arrival of the Mercedes-Benz W222 at dealerships or any wedding involving a star from the sitcom Modern Family. Therefore, I cannot provide an accurate answer based solely on the information provided. The question seems to mix unrelated topics and dates. Could you please specify the date or provide more context?`
        - `Based on the information provided in the context about the 1977 NBA Finals between the Portland Trail Blazers and the Philadelphia 76ers, we cannot definitively identify who was number 3 on the 1976-77 NBA World Champion team, as the Philadelphia 76ers lost the series to the Portland Trail Blazers that year. The Portland Trail Blazers were the champions, but the document does not provide a list of their jersey numbers. To answer this question accurately, we would need a roster of the Portland Trail Blazers with jersey numbers from that specific season.`

13. Auto-rater (final answer)

    The auto-rater uses gpt-4o-mini as the judge ($0.150 / 1M input tokens; $0.600 / 1M output tokens). To rate all 529 data points once, it costs around $0.05.

    The quality of the judge is evaluated on 20 data points from the oracle answers in [google sheet](https://docs.google.com/spreadsheets/d/1LJb7Q-XVFF7z15HYyHOAbxhn4yXovNmppWzfUaTaMdM/edit?usp=sharing).
    
    - code: `code/auto_rater_with_GPT.py`; `code/calc_answer_accuracy.py`
    - data: 
    - Note: I tried with Qwen2.5 7B and 14B models as the auto-rater using the prompt in `utils.py` (`auto_eval_prompt_template`). But the model does not follow the instruction at all. Simple heuristic such as `ground_truth.lower() in qwen_answer.lower()` (i.e., exact match) also does not work because some ground truth answers are sentences. 

14. Progress on final answer evaluation
    - [x] oracle
        - [x] judged
        - [x] rated
        - Result: 58.33%
    - [x] zero-shot
        - [x] judged
        - [x] rated
        - Result: 19.70%
    - [x] bm25
        - [x] judged
        - [x] rated
        - Result: 26.52%
    - [ ] dpr
        - [ ] judged
        - [ ] rated
        - Result: 

15. Retrieval metrics
    - code: `code/retrieval_metrics.py`
    - Node: 
        - The metrics include precision@k, recall@k,, F1@k, and MAP. The k is set to 5 by default. 
        - nDCG@k is not included because there is no relevance score in the ground truth. The ground truth urls are equally relevant.
        - Oracle and zero-shot do not need to be evaluated.

16. Progress on retrieval evaluation 
    - bm25
        - [x] evaluated
        - Result:
            - Recall@5:    0.5283
            - Precision@5: 0.2814
            - F1@5:        0.3603
            - MAP:         0.4576
    - dpr
        - [x] evaluated
        - Result:
            - Rec@5:  0.4632
            - Prec@5: 0.2481
            - F1@5:   0.3169
            - MAP:    0.3761
    - Note: DPR performs worse than bm25 in retrieval. The reason can be that the context is too long for DPR to encode.