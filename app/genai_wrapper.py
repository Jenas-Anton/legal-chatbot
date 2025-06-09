# app/genai_wrapper.py

import google.generativeai as genai

class GeminiSummarizer:
    def __init__(self, api_key: str):
        if not api_key or api_key.strip().lower() == "your_gemini_api_key_here":
            raise ValueError("❌ API key missing or invalid. Check your .env or config.")
        
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
        except Exception as e:
            raise RuntimeError(f"❌ Failed to initialize Gemini model: {e}")


    def executive_summary(self, text: str) -> str:
        return self._call_model(
            f"Summarize the following legal text into an executive summary:\n\n{text}"
        )

    def detailed_summary(self, text: str) -> str:
        return self._call_model(
            f"Provide a detailed summary of the following legal document:\n\n{text}"
        )

    def key_points(self, text: str) -> str:
        return self._call_model(
            f"Extract key points from the following legal text:\n\n{text}"
        )

    def roles_and_parties(self, text: str) -> str:
        return self._call_model(
            f"List all involved roles and parties in this legal text:\n\n{text}"
        )

    def timeline(self, text: str) -> str:
        return self._call_model(
            f"Create a timeline of key events based on the following legal text:\n\n{text}"
        )

    def risk_analysis(self, text: str) -> str:
        return self._call_model(
            f"Analyze and list the potential risks in the following legal document:\n\n{text}"
        )

    def comprehensive_analysis(self, text: str) -> str:
        return self._call_model(
            f"Give a comprehensive legal analysis of this document:\n\n{text}"
        )

    def custom_summary(self, text: str, prompt: str) -> str:
        return self._call_model(f"{prompt.strip()}\n\n{text}")

    def answer_query_with_context(self, uploaded_text: str, query: str, similar_docs_context: str = None) -> str:
        context = uploaded_text.strip()
        if similar_docs_context:
            context += f"\n\nSimilar Documents:\n{similar_docs_context.strip()}"
        prompt = f"Answer the question based on the text below.\nQuestion: {query}\n\nText:\n{context}"
        return self._call_model(prompt)

    def _call_model(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(prompt)
            if hasattr(response, "text"):
                return response.text.strip()
            elif hasattr(response, "parts") and response.parts:
                return response.parts[0].text.strip()
            else:
                return "⚠️ No response text generated."
        except Exception as e:
            return f"❌ Error generating content: {str(e)}"
