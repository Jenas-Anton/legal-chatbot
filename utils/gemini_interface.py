import google.generativeai as genai
import json

def generate_case_summary(facts, judgment, api_key):
    genai.configure(api_key=api_key)
    prompt = f"""
You are a legal expert. Analyze this case and provide a structured summary with specific details.

Facts:
{facts}

Judgment:
{judgment}

Extract and format the following information in JSON format:
{{
    "case_no": "...",
    "year": "...",
    "title": "...",
    "summary": "...",
    "outcome": "[Provide a short label like 'Appeal Dismissed', 'Tenant Wins', etc.]",
    "winner": "[Specify Appellant/Respondent/Tenant/Landlord/Unknown]",
    "loser": "[Specify the opposing party or 'Unknown']",
    "relief_granted": "[Yes/No]",
    "reasoning": "[Provide a concise summary of the court's reasoning]",
    "key_issues": ["List the main legal issues considered"]
}}

Ensure all fields are filled appropriately based on the case content.
"""
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    try:
        # Parse the response as JSON
        result = json.loads(response.text.strip())
        return result
    except json.JSONDecodeError:
        # If JSON parsing fails, return the raw text
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
