import streamlit as st
from config import setup_page, show_sidebar
from utils.dataset_loader import load_combined_datasets
from utils.embedding_search import search_similar_cases
from utils.gemini_interface import generate_case_summary, generate_answer
from utils.case_predictor import CasePredictor
import tempfile
import os
import time

# Set up page
setup_page()

# Sidebar
api_key, num_cases = show_sidebar()

# Initialize case predictor
if 'case_predictor' not in st.session_state and api_key:
    st.session_state.case_predictor = CasePredictor(api_key)

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

col1, col2 = st.columns(2)

with col1:
    analyze_button = st.button("🔍 Analyze Legal Question")
with col2:
    predict_button = st.button("🎯 Predict Case Outcome")

# Handle document upload for prediction
if predict_button:
    st.subheader("📄 Upload Legal Document")
    uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'docx', 'doc'])
    
    if uploaded_file and api_key:
        try:
            # Create a temporary file to save the uploaded content
            with tempfile.NamedTemporaryFile(delete=False, suffix='.' + uploaded_file.name.split('.')[-1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            # Extract text based on file type
            if uploaded_file.name.endswith('.pdf'):
                document_text = CasePredictor.extract_text_from_pdf(tmp_path)
            else:  # doc or docx
                document_text = CasePredictor.extract_text_from_doc(tmp_path)

            # Clean up temporary file
            os.unlink(tmp_path)

            # Get similar cases for context
            similar_cases = search_similar_cases(document_text[:1000], embedder, index, questions, answers, num_cases)
            summarized_cases = [
                {"summary": generate_case_summary(c['question'], c['answer'], api_key), "similarity": c['similarity']}
                for c in similar_cases
            ]

            # Make prediction
            with st.spinner("Analyzing document and making prediction..."):
                prediction_result = st.session_state.case_predictor.predict_case(document_text, summarized_cases)

            # Display results
            st.success("✅ Analysis Complete!")
            
            # Display prediction
            st.subheader("🎯 Case Prediction")
            st.write(f"**Predicted Outcome:** {prediction_result['prediction']}")
            st.write(f"**Confidence:** {prediction_result['confidence']*100:.1f}%")

            # Display feature importance
            st.subheader("📊 Key Factors Influencing Prediction")
            for feature in prediction_result['explanation']['top_features']:
                effect_color = "green" if feature['effect'] == "positive" else "red"
                st.markdown(
                    f"- **{feature['feature']}**: "
                    f"<span style='color:{effect_color}'>{feature['effect'].title()} impact "
                    f"(importance: {abs(feature['importance']):.3f})</span>", 
                    unsafe_allow_html=True
                )

            # Display similar cases
            st.subheader("📚 Similar Cases Referenced")
            for i, case in enumerate(summarized_cases, 1):
                with st.expander(f"Case {i} - Similarity: {case['similarity']:.1f}%"):
                    st.markdown(f'<div class="case-item">{case["summary"]}</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
            if 'tmp_path' in locals():
                try:
                    os.unlink(tmp_path)
                except:
                    pass

    elif not api_key:
        st.error("Please enter your Gemini API key in the sidebar")

# Handle legal question analysis
if analyze_button and query and api_key:
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

elif analyze_button:
    if not query: st.error("Please enter a legal question")
    if not api_key: st.error("Please enter your Gemini API key")

st.markdown("---")
st.markdown("*Disclaimer:* This tool provides general legal information only.\n\n*Data Sources:* Combined Hugging Face legal datasets.")
