import os
import google.generativeai as genai

API_KEY = os.getenv("GOOGLE_API_KEY")  # Make sure this is set in your environment
genai.configure(api_key=API_KEY)

def generate_answer_with_gemini(query, retrieved_qas):
    context = "\n\n".join([f"Q: {qa['question']}\nA: {qa['answer']}" for qa in retrieved_qas])
    
    prompt = f"""You are a legal assistant knowledgeable in Indian law.

Use the following retrieved Q&A as context to answer the user's question:

Context:
{context}

User's question: {query}

Please provide a clear and accurate answer based on the context."""
    
    response = genai.generate_text(
        model="models/text-bison-001",
        prompt=prompt,
        temperature=0.2,
        max_output_tokens=512
    )
    return response.text
