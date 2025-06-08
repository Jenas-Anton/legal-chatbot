import google.generativeai as genai

def generate_case_summary(facts, judgment, api_key):
    genai.configure(api_key=api_key)
    prompt = f"""
Facts:
{facts}

Judgment:
{judgment}

Extract:
📂 Case No: ...
📅 Year: ...
🧑‍⚖️ Title: ...
📝 Summary: ...
"""
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text.strip()

def generate_answer(query, similar_cases, api_key):
    genai.configure(api_key=api_key)
    context = ""
    for i, case in enumerate(similar_cases, 1):
        context += f"Case {i} Facts:\n{case['question']}\nCase {i} Judgment:\n{case['answer']}\n\n"

    prompt = f"""
You are an expert Indian legal assistant.

Question:
{query}

Context:
{context}

Answer:
"""
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text.strip()
