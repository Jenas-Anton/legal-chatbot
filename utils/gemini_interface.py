import google.generativeai as genai

def generate_case_summary(facts, judgment, api_key):
    genai.configure(api_key=api_key)
    prompt = f"""
You are a legal expert. Analyze this case and provide a structured summary with specific details.

Facts:
{facts}

Judgment:
{judgment}

Format your response in clean markdown as follows:

## Case Details
**Case No:** [Extract case number or "Unknown"]
**Year:** [Extract year or "Unknown"] 
**Title:** [Extract case title or create descriptive title]

## Summary
[Brief 2-3 sentence summary of the case]

## Outcome
**Decision:** [Short label like 'Appeal Dismissed', 'Petition Allowed', etc.]
**Winner:** [Appellant/Respondent/Tenant/Landlord/Unknown]
**Loser:** [The opposing party or "Unknown"]
**Relief Granted:** [Yes/No/Partial]

## Court's Reasoning
[Provide a concise summary of the court's reasoning]

## Key Legal Issues
- [Issue 1]
- [Issue 2] 
- [Issue 3]

Ensure all sections are filled appropriately based on the case content.
"""
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text.strip()

def generate_answer(query, similar_cases, api_key):
    genai.configure(api_key=api_key)
    context = ""
    for i, case in enumerate(similar_cases, 1):
        context += f"**Case {i} Facts:**\n{case['question']}\n\n**Case {i} Judgment:**\n{case['answer']}\n\n---\n\n"

    prompt = f"""
You are an expert Indian legal assistant.

**Question:**
{query}

**Similar Cases for Reference:**
{context}

Provide a comprehensive legal analysis in markdown format with the following structure:

## Legal Analysis

### Overview
[Brief overview of the legal issue]

### Applicable Law
[Relevant laws, sections, and provisions]

### Case Analysis
[Analysis based on similar cases provided]

### Legal Precedents
[Reference to relevant precedents if applicable]

### Conclusion & Recommendations
[Your legal opinion and practical advice]

Format your response clearly with proper headings and bullet points where appropriate.
"""
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text.strip()

# Usage in Streamlit
def display_case_summary_markdown(markdown_text):
    """Display the markdown-formatted case summary"""
    import streamlit as st
    st.markdown(markdown_text)

def display_legal_answer(markdown_text):
    """Display the markdown-formatted legal analysis"""
    import streamlit as st
    st.markdown(markdown_text)

