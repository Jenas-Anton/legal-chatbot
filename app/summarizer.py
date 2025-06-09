from app.text_processor import TextProcessor
from app.document_loader import DocumentLoader
from app.query_system import DocumentQuerySystem
from app.genai_wrapper import GeminiSummarizer  # ✅ fixed import

class LegalDocumentSummarizer:
    def __init__(self, api_key: str):
        self.summarizer = GeminiSummarizer(api_key)
        self.processor = TextProcessor()
        self.loader = DocumentLoader()
        self.query_system = DocumentQuerySystem()

    def query_document(self, file_path: str, query: str, k: int = 5) -> str:
        try:
            uploaded_text = self.loader.load_document(file_path)
            similar_docs = self.query_system.get_similar_documents(uploaded_text, k=k)
            similar_docs_context = self.query_system.format_similar_documents_context(similar_docs) if similar_docs else None
            return self.summarizer.answer_query_with_context(
                uploaded_text=uploaded_text,
                query=query,
                similar_docs_context=similar_docs_context
            )
        except Exception as e:
            return f"Error processing query: {e}"

    def summarize_document(self, file_path: str, summary_type: str, custom_prompt: str = None) -> str:
        try:
            text = self.loader.load_document(file_path)
        except Exception as e:
            return f"Error loading document: {e}"

        chunks = self.processor.chunk_text(text)

        if len(chunks) > 1:
            chunk_summaries = [
                f"--- Chunk {i+1} Summary ---\n{self._get_summary(chunk, summary_type, custom_prompt)}"
                for i, chunk in enumerate(chunks)
            ]
            combined = "\n\n".join(chunk_summaries)
            return self._get_summary(combined, summary_type, custom_prompt)
        else:
            return self._get_summary(text, summary_type, custom_prompt)

    def _get_summary(self, text: str, summary_type: str, custom_prompt: str = None) -> str:
        summary_func_map = {
            "executive": self.summarizer.executive_summary,
            "detailed": self.summarizer.detailed_summary,
            "key_points": self.summarizer.key_points,
            "roles_parties": self.summarizer.roles_and_parties,
            "timeline": self.summarizer.timeline,
            "risk_analysis": self.summarizer.risk_analysis,
            "comprehensive": self.summarizer.comprehensive_analysis,
            "custom": lambda text: self.summarizer.custom_summary(text, custom_prompt)
        }
        func = summary_func_map.get(summary_type)
        return func(text) if func else "Invalid summary type selected"
