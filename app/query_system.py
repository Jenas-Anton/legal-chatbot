import faiss
import numpy as np

class DocumentQuerySystem:
    def __init__(self):
        self.index = faiss.IndexFlatL2(768)
        self.documents = []

    def get_similar_documents(self, query_text: str, k: int = 5):
        if not self.documents or self.index.ntotal == 0:
            return []  # No documents or empty index

        query_vector = self._get_vector(query_text)
        try:
            _, indices = self.index.search(np.array([query_vector]), k)
            return [
                self.documents[i]
                for i in indices[0]
                if i < len(self.documents)
            ]
        except Exception as e:
            print(f"⚠️ FAISS search failed: {e}")
            return []

    def format_similar_documents_context(self, docs):
        return "\n---\n".join(docs)

    def _get_vector(self, text: str):
        return np.random.rand(768).astype('float32')
