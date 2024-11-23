import os
import pdb
import json
import faiss
import utils
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from transformers import DPRContextEncoder, DPRQuestionEncoder, \
    DPRContextEncoderTokenizer, DPRQuestionEncoderTokenizer

def encode_in_batches(data, tokenizer, encoder, batch_size=4):
    embeddings = []
    for i in tqdm(range(0, len(data), batch_size)):
        batch = data[i:i + batch_size]
        inputs = tokenizer(batch, truncation=True, padding=True, return_tensors="pt", max_length=512)
        batch_embeddings = encoder(**inputs).pooler_output.detach().numpy()
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

"""
SET UP DPR RETRIEVER
"""
ctx_encoder = DPRContextEncoder.from_pretrained(
    "facebook/dpr-ctx_encoder-single-nq-base"
    )
ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
    "facebook/dpr-ctx_encoder-single-nq-base"
    )
question_encoder = DPRQuestionEncoder.from_pretrained(
    "facebook/dpr-question_encoder-single-nq-base"
    )
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
    "facebook/dpr-question_encoder-single-nq-base"
    )
print("DPR models loaded.")

"""
LOAD DATA
"""
frames_file = "data/frames_dataset_2_5_links_filtered.jsonl"
with open(frames_file, "r") as f:
    frames_data = [json.loads(x.strip()) for x in f.readlines() if x.strip()]
print("Loaded frames_data.")

wiki_file = "data/wikipedia/jsonl_output/wikipedia_filtered.jsonl"
with open(wiki_file, "r") as f:
    wiki_data = [json.loads(x.strip()) for x in f.readlines() if x.strip()]

wiki_data_text = [x["contents"].strip() for x in wiki_data]
questions = [x["Prompt"].strip() for x in frames_data]

wiki_content_url_file = "data/wikipedia/jsonl_output/wikipedia_filtered_content_to_url.json"
with open(wiki_content_url_file, "r") as f:
    wiki_content_url_dict = json.load(f)
print("Loaded wikipedia data.")

"""
EMBEDDING
"""
save_directory = "data/embeddings"
os.makedirs(save_directory, exist_ok=True)

print("Embedding wiki passages...")
wiki_embedding_file = f"{save_directory}/wiki_embeddings.npy"
if os.path.exists(wiki_embedding_file):
    wiki_embeddings = np.load(wiki_embedding_file)
    print("Wiki embeddings loaded from saved files.")
else:
    wiki_embeddings = encode_in_batches(wiki_data_text, ctx_tokenizer, ctx_encoder)
    # save wiki embeddings
    wiki_embeddings_path = wiki_embedding_file
    np.save(wiki_embeddings_path, wiki_embeddings)
    print("Wiki embeddings saved.")

print("Building index...")
index_file = f"{save_directory}/faiss_index.bin"
if os.path.exists(index_file):
    index = faiss.read_index(index_file)
    print("Faiss index loaded from saved files.")
else:
    index = faiss.IndexFlatIP(wiki_embeddings.shape[1])
    # Inner Product (cosine similarity if normalized)
    faiss.normalize_L2(wiki_embeddings)
    print("Adding batches to index...")
    batch_size = 200
    for i in tqdm(range(0, len(wiki_embeddings), batch_size)):
        index.add(wiki_embeddings[i:i + batch_size])
    # save index
    faiss.write_index(index, index_file)
    print("Faiss index saved.")

print("Embedding questions...")
question_embedding_file = f"{save_directory}/question_embeddings.npy"
if os.path.exists(question_embedding_file):
    question_embeddings = np.load(question_embedding_file)
    print("Question embeddings loaded from saved files.")
else:
    question_embeddings = encode_in_batches(questions, question_tokenizer, question_encoder)
    # save question embeddings
    question_embeddings_path = question_embedding_file
    np.save(question_embeddings_path, question_embeddings)
    print("Question embeddings saved.")

faiss.normalize_L2(question_embeddings)

top_k = 5
scores, indices = index.search(question_embeddings, top_k)

print("Writing to file...")
DPR_retrieve_output_file = "data/Qwen_Outputs/naive_rag_baseline_DPR_retrieve.jsonl"
start_point = utils.get_start_point(DPR_retrieve_output_file)

with open(DPR_retrieve_output_file, "w") as f:
    for dict_item in tqdm(frames_data[start_point:]):
        question = dict_item["Prompt"].strip()
        i = questions.index(question)
        top_passages = [
            {"passage": wiki_data_text[idx], "score": score}
            for idx, score in zip(indices[i], scores[i])
        ]

        DPR_retrieve_results = []
        for item in top_passages:
            context_key = item["passage"][:200]
            wiki_link = wiki_content_url_dict.get(context_key, "")
            if not wiki_link:
                print("No wiki_link found for context_key:", context_key)
                pdb.set_trace()
            DPR_retrieve_results.append([wiki_link, float(item["score"])])

        dict_item["naive_rag_retrieve_results_DPR"] = DPR_retrieve_results

        f.write(json.dumps(dict_item) + "\n")
