def search_similar_cases(query, embedder, index, questions, answers, k=5):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(query_embedding.astype("float32"), k)

    results = []
    for i, idx in enumerate(indices[0]):
        similarity = max(0, 100 - (distances[0][i] * 20))
        results.append({
            "question": questions[idx],
            "answer": answers[idx],
            "similarity": similarity
        })

    return results
