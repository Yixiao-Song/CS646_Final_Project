import os
import ast
import json
import faiss
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple
from pyserini.search.lucene import LuceneSearcher
from QwenGeneration import QwenGeneration
from GPTGeneration import GPTGeneration
from transformers import DPRContextEncoder, DPRQuestionEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoderTokenizer
from sklearn.metrics.pairwise import cosine_similarity

query_generation_prompt = """Based on the question and current context, generate {k} different search queries to help find relevant information.
The queries should:
1. Be diverse and not repeat information
2. Build upon the current context
3. Help resolve remaining uncertainties
4. Follow a step-by-step reasoning process

Think carefully about what information is still missing to answer the question.
Plan your search strategy:
1. First, identify key concepts needed from the question
2. Then, look for supporting facts and relationships
3. Finally, seek any missing details or confirmation

Question: {question}
Current Context: {context}

Generate exactly {k} different search queries, separated by newlines:
"""

class NKRetrievalPipeline:
    def __init__(self, retriever_type="bm25", n_steps=5, k_queries=5, n_docs=10):
        self.retriever_type = retriever_type
        self.n_steps = n_steps
        self.k_queries = k_queries
        self.n_docs = n_docs
        self.qwen_model = QwenGeneration()
        self.gpt_model = GPTGeneration(max_tokens=2000)
        
        with open("/project/pi_miyyer_umass_edu/yixiao/CS646/FinalProject/data/wikipedia/jsonl_output/wikipedia_filtered_url_to_content.json", "r") as f:
            self.wiki_url_contents_dict = json.load(f)
        print("Loaded wiki_url_contents_dict.")

        if retriever_type == "bm25":
            self._init_bm25()
        elif retriever_type == "dpr":
            self._init_dpr()
        else:
            raise ValueError(f"Unknown retriever type: {retriever_type}")

        self.query_generation_prompt = query_generation_prompt

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

        # Load wiki data
        with open("data/wikipedia/jsonl_output/wikipedia_filtered.jsonl", "r") as f:
            self.wiki_data = [json.loads(x.strip()) for x in f.readlines() if x.strip()]
        self.wiki_data_text = [x["contents"].strip() for x in self.wiki_data]

        with open("data/wikipedia/jsonl_output/wikipedia_filtered_content_to_url.json", "r") as f:
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

    def _bm25_search(self, query, k=10):
        """Perform BM25 search"""
        hits = self.searcher.search(query, k=k)
        url_score_tuple_lst = []
        for i in range(len(hits)):
            docid = hits[i].docid
            score = hits[i].score
            raw_doc_dict = ast.literal_eval(self.searcher.doc(docid).raw())
            url_score_tuple_lst.append((raw_doc_dict['url'], score))
        return url_score_tuple_lst

    def _dpr_search(self, query, k=10):
        """Perform DPR search"""
        query_embedding = self._encode_in_batches([query], self.question_tokenizer, self.question_encoder)
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            context_key = self.wiki_data_text[idx][:200]
            wiki_link = self.wiki_content_url_dict.get(context_key, "")
            if wiki_link:
                results.append((wiki_link, float(score)))
        return results

    def generate_queries(self, question: str, context: str) -> List[str]:
        """Generate k different search queries using QwenGeneration"""
        prompt = self.query_generation_prompt.format(
            k=self.k_queries,
            question=question,
            context=context if context else "No context available yet."
        )
        response = self.qwen_model.get_response(prompt)
        queries = [q.strip() for q in response.strip().split('\n')]
        print(queries[:self.k_queries])
        return queries[:self.k_queries]

    def prepare_context(self, urls: List[str]) -> str:
        """Prepare context from retrieved documents"""
        context = []
        for url in urls:
            if url in self.wiki_url_contents_dict:
                context.append(self.wiki_url_contents_dict[url]['contents'])
        return "\n\n".join(context)

    def retrieve_documents(self, query: str) -> List[Tuple[str, float]]:
        """Retrieve documents using selected retriever"""
        if self.retriever_type == "bm25":
            return self._bm25_search(query, k=self.n_docs)
        else:
            return self._dpr_search(query, k=self.n_docs)

    def process_question(self, question: str) -> Dict:
        """Process a single question using n-step k-query retrieval"""
        context = ""
        all_retrieved_docs = set()
        retrieval_history = []

        for step in range(self.n_steps):
            step_results = {
                "step": step + 1,
                "queries": [],
                "retrieved_docs": []
            }

            queries = self.generate_queries(question, context)
            
            new_docs = set()
            for query in queries:
                retrieved = self.retrieve_documents(query)
                step_results["queries"].append(query)
                step_results["retrieved_docs"].append(retrieved)
                
                for url, score in retrieved:
                    if url not in all_retrieved_docs:
                        new_docs.add(url)
                        all_retrieved_docs.add(url)
            
            new_context = self.prepare_context(list(new_docs))
            context = f"{context}\n\n{new_context}" if context else new_context
            
            retrieval_history.append(step_results)

        return {
            "question": question,
            "final_context": context,
            "retrieval_history": retrieval_history
        }

    def process_dataset(self, input_file: str, output_file: str):
        """Process entire dataset"""
        with open(input_file, "r") as f:
            data = [json.loads(line) for line in f]
        
        for item in tqdm(data):
            question = item["Prompt"]
            result = self.process_question(question)
            
            output_item = {**item, **result}
            with open(output_file, "a") as f:
                f.write(json.dumps(output_item) + "\n")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retriever",
        type=str,
        default="bm25",
        choices=["bm25", "dpr"],
        help="Retriever type to use (bm25, dpr)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5,
        help="Number of steps for n-step retrieval"
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=5,
        help="Number of queries to generate at each step"
    )
    parser.add_argument(
        "--docs",
        type=int,
        default=10,
        help="Number of documents to retrieve at each step"
    )
    args = parser.parse_args()

    pipeline = NKRetrievalPipeline(
        retriever_type=args.retriever,
        n_steps=args.steps,
        k_queries=args.queries,
        n_docs=args.docs
    )
    
    input_file = "/project/pi_miyyer_umass_edu/yixiao/CS646/FinalProject/data/frames_dataset_2_5_links_filtered.jsonl"
    output_file = f"data/Qwen_Outputs/nk_{args.retriever}_step_{args.steps}_queries_{args.queries}_docs_{args.docs}_output.jsonl"
    pipeline.process_dataset(input_file, output_file)

if __name__ == "__main__":
    main()
