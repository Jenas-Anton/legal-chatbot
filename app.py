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

# Create tabs for different functionalities
tab1, tab2 = st.tabs(["📝 Legal Question Analysis", "📄 Case Prediction"])

with tab1:
    st.subheader("💬 Ask Your Legal Question")
    query = st.text_area("Enter your legal question or scenario", height=150)
    analyze_button = st.button("Analyze Question")

    if analyze_button and query and api_key:
        start_time = time.time()
        with st.spinner("Searching legal database..."):
            similar_cases = search_similar_cases(query, index, questions, answers, num_cases)

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

with tab2:
    st.subheader("📄 Upload Case Document")
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
            with st.spinner("Finding similar cases..."):
                similar_cases = search_similar_cases(document_text[:1000], index, questions, answers, num_cases)
                summarized_cases = [
                    {"summary": generate_case_summary(c['question'], c['answer'], api_key), "similarity": c['similarity']}
                    for c in similar_cases
                ]

            # Make prediction
            with st.spinner("Analyzing document and making prediction..."):
                prediction_result = st.session_state.case_predictor.predict_case(document_text, summarized_cases)

            # Display results
            st.subheader("📊 Prediction Results")
            
            # Prediction box with color based on confidence
            confidence = prediction_result['confidence']
            color = "green" if confidence > 0.7 else "orange" if confidence > 0.5 else "red"
            st.markdown(f"""
                <div style="padding: 1rem; border-radius: 0.5rem; border-left: 5px solid {color}; background-color: {color}10">
                    <h3 style="color: {color}">Prediction: {prediction_result['prediction']}</h3>
                    <p>Confidence: {confidence:.1%}</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Show key factors
            if prediction_result['explanation']['top_features']:
                st.subheader("🔑 Key Factors")
                for feature in prediction_result['explanation']['top_features']:
                    effect = "Supporting" if feature['effect'] == "positive" else "Opposing"
                    bg_color = "#90EE90" if feature['effect'] == "positive" else "#FFB6C1"  # Light green or light red
                    border_color = "#228B22" if feature['effect'] == "positive" else "#DC143C"  # Dark green or crimson
                    st.markdown(f"""
                        <div style="padding: 0.75rem; margin: 0.5rem 0; border-radius: 0.5rem; background-color: {bg_color}; border: 2px solid {border_color}">
                            <strong>{feature['feature']}</strong>: {effect} factor (weight: {abs(feature['importance']):.4f})
                        </div>
                    """, unsafe_allow_html=True)
            
            # Show all similar cases
            st.subheader("📚 Similar Cases")
            st.write(f"Showing all {len(summarized_cases)} similar cases used in analysis:")
            for i, case in enumerate(summarized_cases, 1):
                with st.expander(f"Case {i} - Similarity: {case['similarity']:.1f}%"):
                    st.markdown(f'<div class="case-item">{case["summary"]}</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    elif uploaded_file:
        st.error("Please enter your Gemini API key")

st.markdown("---")
st.markdown("*Disclaimer:* This tool provides general legal information only.\n\n*Data Sources:* Combined Hugging Face legal datasets.")
