import streamlit as st
import tempfile
import os
import time
from pathlib import Path

# --- Page Setup (MUST BE FIRST) ---
st.set_page_config(
    page_title="Verdict AI", 
    page_icon="⚖️",
    layout="wide"
)

# Custom modules (make sure they are in your project)
from app.summarizer import LegalDocumentSummarizer
from app.utils import save_uploaded_file
from config import show_sidebar  # Remove setup_page import
from utils.dataset_loader import load_combined_datasets
from utils.embedding_search import search_similar_cases
from utils.gemini_interface import generate_case_summary, generate_answer
from utils.case_predictor import CasePredictor

# Import tab modules
from tabs.legal_analysis import show_legal_analysis_tab
from tabs.case_prediction import show_case_prediction_tab
from tabs.document_summary import show_document_summary_tab
from tabs.document_generator import show_document_generator_tab

# Main title and description
st.title("⚖️ Verdict AI : Legal AI Assistant")
st.markdown("""
Welcome to your all-in-one legal AI platform. This application provides:
- **Legal Question Analysis** with case similarity search
- **Case Outcome Prediction** from uploaded documents
- **Document Summarization & Q&A** capabilities
- **Legal Document Generation** with AI-powered clause creation
""")

# Sidebar setup
api_key, num_cases = show_sidebar()

# API key session state
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""
if api_key != st.session_state.api_key:
    st.session_state.api_key = api_key
    if 'assistant' in st.session_state:
        del st.session_state.assistant
    if 'case_predictor' in st.session_state:
        del st.session_state.case_predictor

# Check for API key
if not api_key or api_key.strip() == "":
    st.warning("🔑 Please enter your Gemini API key in the sidebar to continue.")
    st.stop()

# Initialize assistants
if 'assistant' not in st.session_state:
    st.session_state.assistant = LegalDocumentSummarizer(api_key)

if 'case_predictor' not in st.session_state:
    st.session_state.case_predictor = CasePredictor(api_key)

# Load Datasets (only for analysis features)
@st.cache_data
def load_legal_datasets():
    try:
        questions, answers, embedder, index, dataset_info = load_combined_datasets()
        return questions, answers, embedder, index, dataset_info, True
    except Exception as e:
        st.error(f"❌ Failed to load datasets: {str(e)}")
        return None, None, None, None, None, False

questions, answers, embedder, index, dataset_info, datasets_loaded = load_legal_datasets()

if datasets_loaded:
    st.success(f"✅ Loaded {len(questions):,} legal cases for analysis")
    with st.expander("📊 Dataset Info"):
        for info in dataset_info:
            st.write(f"**{info['name']}:** {info['count']:,} cases")

# Create shared context for tabs
tab_context = {
    'api_key': api_key,
    'num_cases': num_cases,
    'questions': questions,
    'answers': answers,
    'embedder': embedder,
    'index': index,
    'dataset_info': dataset_info,
    'datasets_loaded': datasets_loaded,
    'assistant': st.session_state.assistant,
    'case_predictor': st.session_state.case_predictor
}

# --- Main Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📝 Legal Question Analysis",
    "📄 Case Prediction", 
    "📋 Document Summarization & Query",
    "📃 Legal Document Generator"
])

# Call individual tab functions
with tab1:
    show_legal_analysis_tab(tab_context)

with tab2:
    show_case_prediction_tab(tab_context)

with tab3:
    show_document_summary_tab(tab_context)

with tab4:
    show_document_generator_tab()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p><strong>⚠️ Legal Disclaimer:</strong> This tool provides general legal information and document templates for educational purposes only. 
    Generated content should be reviewed by qualified legal professionals before use in any legal context.</p>
    <p><strong>📊 Data Sources:</strong> Combined Hugging Face legal datasets | <strong>🤖 AI Models:</strong> Gemini API</p>
    <p>© 2024 Comprehensive Legal AI Assistant</p>
</div>
""", unsafe_allow_html=True)