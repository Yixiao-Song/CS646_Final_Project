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

rerank_prompt_template = """Given a query and a set of retrieved documents for this query, select the top five most relevant documents. Return the indices of the documents in the order of their relevance, separated by commas. Do not return anything else.

Query: {question}

Retrieved Documents:

{context}

The indices of the five documents most relevant to the query are as follows, separated by commas:"""

class RetrievalPipeline:
    def __init__(self, retriever_type="bm25", maxq=5, topk=2, ndocs=5, bsz=1, use_reranking=False):
        self.retriever_type = retriever_type
        self.qwen_model = QwenGeneration()
        self.rerank_prompt_template = rerank_prompt_template
        self.use_reranking = use_reranking
        
        with open("/project/pi_miyyer_umass_edu/yixiao/CS646/FinalProject/data/wikipedia/jsonl_output/wikipedia_filtered_url_to_content.json", "r") as f:
            self.wiki_url_contents_dict = json.load(f)
        print("Loaded wiki_url_contents_dict.")

        if retriever_type == "bm25":
            self._init_bm25()
        elif retriever_type == "dpr":
            self._init_dpr()
        else:
            raise ValueError(f"Unknown retriever type: {retriever_type}")
        
        self.maxq = maxq
        self.topk = topk
        self.ndocs = ndocs
        self.bsz = bsz
    
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

        with open("/project/pi_miyyer_umass_edu/yixiao/CS646/FinalProject/data/wikipedia/jsonl_output/wikipedia_filtered.jsonl", "r") as f:
            self.wiki_data = [json.loads(x.strip()) for x in f.readlines() if x.strip()]
        self.wiki_data_text = [x["contents"].strip() for x in self.wiki_data]

        with open("/project/pi_miyyer_umass_edu/yixiao/CS646/FinalProject/data/wikipedia/jsonl_output/wikipedia_filtered_content_to_url.json", "r") as f:
            self.wiki_content_url_dict = json.load(f)

        self._setup_dpr_embeddings()

    def _setup_dpr_embeddings(self):
        """Set up DPR embeddings and FAISS index"""
        save_directory = "/project/pi_miyyer_umass_edu/yixiao/CS646/FinalProject/data/embeddings"
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

    def get_retrieval_key(self):
        """Get the appropriate key for storing retrieval results"""
        if not self.use_reranking:
            return f"Qwen_decomp_{self.retriever_type}_direct_links"
        return f"Qwen_decomp_{self.retriever_type}_reranked_links"

    def get_answer_key(self):
        """Get the appropriate key for storing generated answers"""
        if not self.use_reranking:
            return f"Qwen_decomp_{self.retriever_type}_direct_answer"
        return f"Qwen_decomp_{self.retriever_type}_reranked_answer"

    def process_retrieval(self, frames_data, output_file):
        """Process initial retrieval using selected method"""
        print(f"Starting retrieval process using {self.retriever_type}...")
        if os.path.exists(output_file):
            print(f"Output file {output_file} already exists. Skipping retrieval.")
            return output_file
        with open(output_file, "w") as f:
            for dict_item in tqdm(frames_data):
                combined_results = {}
                queries = dict_item["extracted_subquery"] if "extracted_subquery" in dict_item else [dict_item["Prompt"]]
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
                key_name = f"decomp_rag_retrieve_results_{self.retriever_type}"
                dict_item[key_name] = combined_results
                f.write(json.dumps(dict_item) + "\n")
        return output_file

    def get_direct_links(self, dict_item):
        """Get top links directly from retrieval results without reranking.
    
        Merges duplicate links by taking their maximum score across all subqueries,
        then sorts by score from highest to lowest, returning the top ndocs results.
        """
        key_to_links = f"decomp_rag_retrieve_results_{self.retriever_type}"
        all_links = []
        
        retrieval_results = dict_item[key_to_links]
        for query_results in retrieval_results.values():
            results = query_results['retrieve_results']
            all_links.extend([[link, score] for link, score in results])
        
        link_scores = {}
        for link, score in all_links:
            if link in link_scores:
                # Keep the maximum score
                link_scores[link] = max(link_scores[link], score)
            else:
                link_scores[link] = score
        
        merged_links = [[link, score] for link, score in link_scores.items()]
        sorted_links = sorted(merged_links, key=lambda x: x[1], reverse=True)
        return sorted_links[:self.ndocs]
    
    def process_reranking(self, input_file, output_file):
        """Step 2: Process reranking"""
        print("Starting reranking process...")
        with open(input_file, "r") as f:
            frames_data = [json.loads(x.strip()) for x in f.readlines() if x.strip()]

        start_point = utils.get_start_point(output_file)

        with open(output_file, "a") as f:
            for dict_item in tqdm(frames_data[start_point:]):
                key_to_links = f"decomp_rag_retrieve_results_{self.retriever_type}"
                context, links = utils.prepare_context(
                    dict_item,
                    self.wiki_url_contents_dict,
                    key_to_links=key_to_links,
                    return_links=True,
                    topk=self.topk
                )
                query = dict_item["Prompt"]
                prompt = self.rerank_prompt_template.format(
                    question=query,
                    context=context
                )

                valid = False
                attempts = 0
                while not valid and attempts < 3:
                    response = self.qwen_model.get_response(prompt, max_new_tokens=100)
                    print(f"Qwen reranking response: {response}")
                    valid, indices = utils.find_integer_list(response, n=self.ndocs)
                    attempts += 1
                    print(f"Invalidate response for query: {response}. Attempt {attempts}.")
                if not valid:
                    print(f"Failed to get a valid response after 3 attempts for query: {query}")
                    indices = []
                
                final_links = [links[idx] for idx in indices]

                result_key = f"Qwen_decomp_{self.retriever_type}_reranked"
                dict_item[result_key] = final_links
                f.write(json.dumps(dict_item) + "\n")
        return output_file
    
    def process_generation(self, input_file, output_file, bsz=1):
        """Step 3: Process generation with batch support
        
        Args:
            input_file (str): Path to input file containing retrieval results
            output_file (str): Path to output file for saving generated answers
            bsz (int): Batch size for generation. If 1, processes sequentially
        """
        print(f"Starting generation process with batch size {bsz}...")
        with open(input_file, "r") as f:
            frames_data = [json.loads(x.strip()) for x in f.readlines() if x.strip()]

        # Find indices of items that need processing
        indices_to_process = []
        result_key = self.get_answer_key()
        for idx, item in enumerate(frames_data):
            if result_key not in item:
                indices_to_process.append(idx)
        
        print(f"Items to process: {len(indices_to_process)}. Starts with {indices_to_process[0]}")
        
        if not indices_to_process:
            print("All items already processed. Skipping generation...")
            return

        # Process in batches
        for batch_start in tqdm(range(0, len(indices_to_process), bsz)):
            batch_indices = indices_to_process[batch_start:batch_start + bsz]
            batch_prompts = []
            
            # Prepare all prompts in the batch
            for idx in batch_indices:
                dict_item = frames_data[idx]
                key_to_links = f"Qwen_decomp_{self.retriever_type}_direct_links"
                context, _ = utils.prepare_context(
                    dict_item,
                    self.wiki_url_contents_dict,
                    key_to_links=key_to_links,
                    return_links=True,
                    topk=None
                )
                query = dict_item["Prompt"]
                prompt = utils.oracle_prompt_template.format(
                    context=context,
                    question=query
                )
                batch_prompts.append(prompt)
            
            # Generate responses for the batch
            if bsz == 1:
                responses = [self.qwen_model.get_response(batch_prompts[0], max_new_tokens=512)]
            else:
                responses = self.qwen_model.get_response(batch_prompts, max_new_tokens=512)
            
            # Save responses and write updated data
            for idx, response in zip(batch_indices, responses):
                frames_data[idx][result_key] = response.strip()
                
                # Print debug information
                ground_truth_ans = frames_data[idx]["Answer"]
                print(f"\nProcessing item {idx}:")
                print(f"Ground truth answer: {ground_truth_ans}")
                print(f"Qwen response: {response}")
            
            # Write after each batch to save progress
            with open(output_file, "w") as f:
                for item in frames_data:
                    f.write(json.dumps(item) + "\n")

    def process_pipeline(self, frames_data, base_output_path):
        """Process the entire pipeline with optional reranking"""
        retrieve_output = f"{base_output_path}/decomp_{self.retriever_type}_retrieve_maxq{self.maxq}.jsonl"
        final_output = f"{base_output_path}/decomp_{self.retriever_type}_answers_maxq{self.maxq}_topk{self.topk}_ndocs{self.ndocs}.jsonl"

        # Step 1: Retrieval
        retrieve_file = self.process_retrieval(frames_data, retrieve_output)

        if self.use_reranking:
            # Step 2: Reranking
            self.process_reranking(retrieve_file, final_output)
        else:
            # Skip reranking and directly use top retrieved documents
            print("Skipping reranking, using top retrieved documents directly...")
            if not os.path.exists(os.path.dirname(final_output)):
                os.makedirs(os.path.dirname(final_output))
            
            with open(retrieve_file, 'r') as f:
                frames_data = [json.loads(line.strip()) for line in f]
            
            links_exist = False
            if os.path.exists(final_output):
                with open(final_output, 'r') as f:
                    first_line = f.readline()
                    if self.get_retrieval_key() in json.loads(first_line):
                        print(f"Final output file {final_output} already exists and contains {self.get_retrieval_key()}. Skipping.")
                        links_exist = True
            
            if not links_exist:
                with open(final_output, 'w') as f:
                    for dict_item in frames_data:
                        final_links = self.get_direct_links(dict_item)
                        dict_item[self.get_retrieval_key()] = final_links
                        f.write(json.dumps(dict_item) + "\n")

        # Step 3: Generation with batch size
        self.process_generation(final_output, final_output, bsz=self.bsz)
        
        return final_output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retriever",
        type=str,
        default="bm25",
        choices=["bm25", "dpr"],
        help="Retriever type to use (bm25, dpr)"
    )
    parser.add_argument(
        "--maxq",
        type=int,
        default=None,
        help="Maximum number of queries to generate"
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=None,
        help="Top k documents to retrieve"
    )
    parser.add_argument(
        "--ndocs",
        type=int,
        default=None,
        help="Number of documents to use for answer generation"
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Whether to use reranking"
    )
    parser.add_argument(
        "--bsz",
        type=int,
        default=1,
        help="Batch size for generation"
    )
    args = parser.parse_args()

    pipeline = RetrievalPipeline(
        retriever_type=args.retriever,
        maxq=args.maxq,
        topk=args.topk,
        ndocs=args.ndocs,
        bsz=args.bsz,
        use_reranking=args.rerank
    )
    
    frames_file = f"data/frames_dataset_2_5_links_filtered_extracted_queries_max_{args.maxq}.jsonl"
    print(f"Loading decomposed frames_data from {frames_file}...")
    with open(frames_file, "r") as f:
        frames_data = [json.loads(x.strip()) for x in f.readlines() if x.strip()]
    print("Loaded decomposed frames_data.")

    base_path = "data/Qwen_Outputs"
    pipeline.process_pipeline(frames_data, base_path)

if __name__ == "__main__":
    main()
