from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load dataset once
dataset = load_dataset("Raghvender/IndianLawDataset", split="train")
questions = dataset["question"]
answers = dataset["answer"]

# Load embedding model once
embedder = SentenceTransformer("law-ai/InLegalBERT")

# Embed all questions and build FAISS index
print("Embedding questions...")
question_embeddings = embedder.encode(questions, convert_to_numpy=True, show_progress_bar=True)

dimension = question_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(question_embeddings)

print("FAISS index built.")

def retrieve_similar_questions(query, k=3):
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)
    results = []
    for idx in indices[0]:
        results.append({
            "question": questions[idx],
            "answer": answers[idx]
        })
    return results
