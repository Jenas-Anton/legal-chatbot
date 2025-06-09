# app/utils.py
import tempfile
from pathlib import Path


def save_uploaded_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        import streamlit as st
        st.error(f"Error saving uploaded file: {e}")
        return None