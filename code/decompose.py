import os
import ast
import pdb
import json
import faiss
import utils
import numpy as np
import argparse
from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher
from QwenGeneration import QwenGeneration
from transformers import DPRContextEncoder, DPRQuestionEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoderTokenizer
from sklearn.metrics.pairwise import cosine_similarity

class RetrievalPipeline:
    def __init__(self, retriever_type="bm25"):
        self.retriever_type = retriever_type
        self.qwen_model = QwenGeneration()
        
        with open("/project/pi_miyyer_umass_edu/yixiao/CS646/FinalProject/data/wikipedia/jsonl_output/wikipedia_filtered_url_to_content.json", "r") as f:
            self.wiki_url_contents_dict = json.load(f)
        print("Loaded wiki_url_contents_dict.")

        if retriever_type == "bm25":
            self._init_bm25()
        elif retriever_type == "dpr":
            self._init_dpr()
        else:
            raise ValueError(f"Unknown retriever type: {retriever_type}")

    def _init_bm25(self):
        """Initialize BM25 retriever"""
        self.searcher = LuceneSearcher('/project/pi_miyyer_umass_edu/yixiao/CS646/FinalProject/data/wikipedia/index_output_filtered_wiki')
        print("BM25 retriever initialized.")

    def _init_dpr(self):
        """Initialize DPR retriever and load necessary data"""
        self.ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
        self.ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
        self.question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        self.question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        print("DPR models loaded.")

        with open("data/wikipedia/jsonl_output/wikipedia_filtered.jsonl", "r") as f:
            self.wiki_data = [json.loads(x.strip()) for x in f.readlines() if x.strip()]
        self.wiki_data_text = [x["contents"].strip() for x in self.wiki_data]

        with open("data/wikipedia/jsonl_output/wikipedia_filtered_content_to_url.json", "r") as f:
            self.wiki_content_url_dict = json.load(f)

        self._setup_dpr_embeddings()

    def _setup_dpr_embeddings(self):
        """Set up DPR embeddings and FAISS index"""
        save_directory = "data/embeddings"
        os.makedirs(save_directory, exist_ok=True)

        wiki_embedding_file = f"{save_directory}/wiki_embeddings.npy"
        if os.path.exists(wiki_embedding_file):
            self.wiki_embeddings = np.load(wiki_embedding_file)
            print("Wiki embeddings loaded from saved files.")
        else:
            self.wiki_embeddings = self._encode_in_batches(self.wiki_data_text, self.ctx_tokenizer, self.ctx_encoder)
            np.save(wiki_embedding_file, self.wiki_embeddings)
            print("Wiki embeddings saved.")

        index_file = f"{save_directory}/faiss_index.bin"
        if os.path.exists(index_file):
            self.index = faiss.read_index(index_file)
            print("Faiss index loaded from saved files.")
        else:
            self.index = faiss.IndexFlatIP(self.wiki_embeddings.shape[1])
            faiss.normalize_L2(self.wiki_embeddings)
            print("Adding batches to index...")
            batch_size = 200
            for i in tqdm(range(0, len(self.wiki_embeddings), batch_size)):
                self.index.add(self.wiki_embeddings[i:i + batch_size])
            faiss.write_index(self.index, index_file)
            print("Faiss index saved.")

    def _encode_in_batches(self, data, tokenizer, encoder, batch_size=4):
        """Encode text in batches using DPR"""
        embeddings = []
        for i in tqdm(range(0, len(data), batch_size)):
            batch = data[i:i + batch_size]
            inputs = tokenizer(batch, truncation=True, padding=True, return_tensors="pt", max_length=512)
            batch_embeddings = encoder(**inputs).pooler_output.detach().numpy()
            embeddings.append(batch_embeddings)
        return np.vstack(embeddings)

    def _bm25_search(self, query, k=5):
        """Perform BM25 search"""
        hits = self.searcher.search(query, k=k)
        url_score_tuple_lst = []
        for i in range(len(hits)):
            docid = hits[i].docid
            score = hits[i].score
            raw_doc_dict = ast.literal_eval(self.searcher.doc(docid).raw())
            url_score_tuple_lst.append((raw_doc_dict['url'], score))
        return url_score_tuple_lst

    def _dpr_search(self, query, k=5):
        """Perform DPR search"""
        query_embedding = self._encode_in_batches([query], self.question_tokenizer, self.question_encoder)
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            context_key = self.wiki_data_text[idx][:200]
            wiki_link = self.wiki_content_url_dict.get(context_key, "")
            if not wiki_link:
                print("No wiki_link found for context_key:", context_key)
                pdb.set_trace()
            results.append((wiki_link, float(score)))
        return results

    def process_retrieval(self, frames_data, output_file):
        """Process initial retrieval using selected method"""
        print(f"Starting retrieval process using {self.retriever_type}...")
        with open(output_file, "w") as f:
            for dict_item in tqdm(frames_data):
                combined_results = {}
                queries = dict_item["extracted_queries"] if "extracted_queries" in dict_item else [dict_item["Prompt"]]
                for query_idx, query in enumerate(queries):
                    print(f"Processing query {query_idx + 1}: {query}")
                    if self.retriever_type == "bm25":
                        results = self._bm25_search(query)
                    else:
                        results = self._dpr_search(query)
                    combined_results[f"query_{query_idx + 1}"] = {
                        "query_text": query,
                        "retrieve_results": results
                    }
                key_name = "decomp_rag_retrieve_results_BM25" if self.retriever_type == "bm25" else "decomp_rag_retrieve_results_DPR"
                dict_item[key_name] = combined_results
                f.write(json.dumps(dict_item) + "\n")
        return output_file

    # Rest of the methods remain the same as in previous version
    def process_reranking(self, input_file, output_file):
        """Step 2: Process reranking"""
        print("Starting reranking process...")
        with open(input_file, "r") as f:
            frames_data = [json.loads(x.strip()) for x in f.readlines() if x.strip()]

        start_point = utils.get_start_point(output_file)

        with open(output_file, "a") as f:
            for dict_item in tqdm(frames_data[start_point:]):
                key_to_links = 'decomp_rag_retrieve_results_BM25' if self.retriever_type == "bm25" else 'decomp_rag_retrieve_results_DPR'
                context, links = utils.prepare_context_rerank(
                    dict_item,
                    self.wiki_url_contents_dict,
                    key_to_links=key_to_links
                )
                query = dict_item["Prompt"]
                prompt = utils.rerank_prompt_template.format(
                    question=query,
                    context=context
                )

                valid = False
                attempts = 0
                while not valid and attempts < 5:
                    response = self.qwen_model.get_response(prompt)
                    valid = utils.is_integer_list(response)
                    attempts += 1
                if not valid:
                    print(f"Failed to get a valid response after 5 attempts for query: {query}")
                    pdb.set_trace()

                result_key = "Qwen_decomp_BM25_reranked" if self.retriever_type == "bm25" else "Qwen_decomp_DPR_reranked"
                dict_item[result_key] = [item.strip() for item in response.split(',')]
                f.write(json.dumps(dict_item) + "\n")
        return output_file

    def process_generation(self, input_file, output_file):
        """Step 3: Process generation"""
        print("Starting generation process...")
        with open(input_file, "r") as f:
            frames_data = [json.loads(x.strip()) for x in f.readlines() if x.strip()]

        start_point = utils.get_start_point(output_file)

        with open(output_file, "a") as f:
            for dict_item in tqdm(frames_data[start_point:]):
                key_to_links = "Qwen_decomp_BM25_reranked" if self.retriever_type == "bm25" else "Qwen_decomp_DPR_reranked"
                context = utils.prepare_context(
                    dict_item,
                    self.wiki_url_contents_dict,
                    key_to_links=key_to_links
                )
                query = dict_item["Prompt"]
                prompt = utils.oracle_prompt_template.format(
                    context=context,
                    question=query
                )

                response = self.qwen_model.get_response(prompt)
                ground_truth_ans = dict_item["Answer"]

                print(f"Ground truth answer: {ground_truth_ans}")
                print(f"Qwen response: {response}")

                result_key = "Qwen_decomp_BM25_answer" if self.retriever_type == "bm25" else "Qwen_decomp_DPR_answer"
                dict_item[result_key] = response.strip()
                f.write(json.dumps(dict_item) + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retriever",
        type=str,
        default="bm25",
        choices=["bm25", "dpr"],
        help="Retriever type to use (bm25, dpr)"
    )
    args = parser.parse_args()

    pipeline = RetrievalPipeline(retriever_type=args.retriever)
    
    frames_file = "data/frames_dataset_2_5_links_filtered_decomposed.jsonl"
    with open(frames_file, "r") as f:
        frames_data = [json.loads(x.strip()) for x in f.readlines() if x.strip()]
    print("Loaded decomposed frames_data.")

    base_path = "data/Qwen_Outputs"
    if args.retriever == "bm25":
        retrieve_output = "data/decomp_BM25_retrieve.jsonl"
        rerank_output = f"{base_path}/decomp_BM25_output.jsonl"
        generate_output = f"{base_path}/decomp_BM25_output.jsonl"
    elif args.retriever == "dpr":
        retrieve_output = f"data/decomp_DPR_retrieve.jsonl"
        rerank_output = f"{base_path}/decomp_DPR_output.jsonl"
        generate_output = f"{base_path}/decomp_DPR_output.jsonl"
    else:
        raise ValueError(f"Unknown retriever type: {args.retriever}")

    retrieve_file = pipeline.process_retrieval(frames_data, retrieve_output)
    rerank_file = pipeline.process_reranking(retrieve_file, rerank_output)
    pipeline.process_generation(rerank_file, generate_output)

if __name__ == "__main__":
    main()
