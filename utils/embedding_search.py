import numpy as np
from sentence_transformers import SentenceTransformer

def get_embedder():
    """Get the sentence transformer model for embeddings"""
    return SentenceTransformer('all-MiniLM-L6-v2')

def search_similar_cases(query, index, questions, answers, k=5):
    """Search for similar cases using FAISS index"""
    try:
        # Get embedder
        embedder = get_embedder()
        
        # Get query embedding
        query_embedding = embedder.encode([query], convert_to_tensor=True).cpu().numpy()
        
        # Search in FAISS index
        D, I = index.search(query_embedding, k)
        
        # Format results
        results = []
        for i, (dist, idx) in enumerate(zip(D[0], I[0])):
            similarity = 1 - dist  # Convert distance to similarity score
            results.append({
                'question': questions[idx],
                'answer': answers[idx],
                'similarity': similarity * 100  # Convert to percentage
            })
            
        return results
        
    except Exception as e:
        print(f"Error in search_similar_cases: {str(e)}")
        return []
