import streamlit as st

def setup_page():
    st.set_page_config(page_title="Indian Legal Assistant", page_icon="⚖️", layout="wide")
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">⚖️ Indian Legal Assistant</h1>', unsafe_allow_html=True)

def show_sidebar():
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("Google Gemini API Key", type="password")
        num_cases = st.slider("Similar cases to analyze", 3, 10, 5)
    return api_key, num_cases
