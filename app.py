import streamlit as st
from config import setup_page, show_sidebar
from utils.dataset_loader import load_combined_datasets
from utils.embedding_search import search_similar_cases
from utils.gemini_interface import generate_case_summary, generate_answer

import time

# Set up page
setup_page()

# Sidebar
api_key, num_cases = show_sidebar()

# Load datasets
try:
    with st.spinner("Loading legal databases..."):
        questions, answers, embedder, index, dataset_info = load_combined_datasets()
    st.success(f"✅ Successfully loaded {len(questions):,} total cases")

    with st.expander("📊 Dataset Information"):
        for info in dataset_info:
            st.write(f"*{info['name']}:* {info['count']:,} cases")

except Exception as e:
    st.error(f"❌ Error loading datasets: {str(e)}")
    st.stop()

# Query interface
st.subheader("💬 Ask Your Legal Question")
examples = [
    "What are the grounds for divorce under Hindu Marriage Act?",
    "How to file for anticipatory bail in 498A case?",
    "What are consumer rights under Consumer Protection Act?",
    "Property dispute resolution in India",
    "Rights of accused during police custody"
]

query = st.text_area("Enter your legal question:", height=100, placeholder="e.g., What are the legal remedies for breach of contract?")
for example in examples:
    if st.button(example):
        query = example
        st.experimental_rerun()

if st.button("🔍 Analyze Legal Question") and query and api_key:
    start_time = time.time()
    with st.spinner("Searching legal database..."):
        similar_cases = search_similar_cases(query, embedder, index, questions, answers, num_cases)

    with st.spinner("Generating summaries..."):
        summarized_cases = [
            {"summary": generate_case_summary(c['question'], c['answer'], api_key), "similarity": c['similarity']}
            for c in similar_cases
        ]

    with st.spinner("Generating analysis..."):
        analysis = generate_answer(query, similar_cases, api_key)

    st.success(f"✅ Completed in {time.time() - start_time:.2f} seconds")
    st.subheader("📋 Legal Analysis")
    st.markdown(f'<div class="answer-box">{analysis}</div>', unsafe_allow_html=True)

    st.subheader("📚 Similar Cases Found (Summarized)")
    for i, case in enumerate(summarized_cases, 1):
        with st.expander(f"Case {i} - Similarity: {case['similarity']:.1f}%"):
            st.markdown(f'<div class="case-item">{case["summary"]}</div>', unsafe_allow_html=True)

elif st.button("🔍 Analyze Legal Question"):
    if not query: st.error("Please enter a legal question")
    if not api_key: st.error("Please enter your Gemini API key")

st.markdown("---")
st.markdown("*Disclaimer:* This tool provides general legal information only.\n\n*Data Sources:* Combined Hugging Face legal datasets.")
