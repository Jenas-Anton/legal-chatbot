from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
import streamlit as st
import numpy as np
from utils.extractors import extract_case_parts
import pickle
import os
import re

@st.cache_resource
def load_combined_datasets():
    """Load datasets from local files if available, otherwise from Hugging Face"""
    # Check for cached files in the parent directory
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    questions_path = os.path.join(parent_dir, "questions.pkl")
    answers_path = os.path.join(parent_dir, "answers.pkl")
    index_path = os.path.join(parent_dir, "faiss_index.index")
    
    # Try loading from local files first
    if (os.path.exists(questions_path) and 
        os.path.exists(answers_path) and 
        os.path.exists(index_path)):
        
        st.info("📂 Loading cached datasets from local files...")
        
        # Load questions and answers
        with open(questions_path, "rb") as f:
            questions = pickle.load(f)
        with open(answers_path, "rb") as f:
            answers = pickle.load(f)
            
        # Load FAISS index
        index = faiss.read_index(index_path)
        
        # Load sentence transformer
        embedder = SentenceTransformer("law-ai/InLegalBERT")
        
        st.success("✅ Successfully loaded cached datasets")
        return questions, answers, embedder, index, [{"name": "Cached Dataset", "count": len(questions)}]
    
    # If local files not found, load from Hugging Face
    st.warning("⚠️ Local files not found. Loading from Hugging Face...")
    
    datasets = [
        "santoshtyss/indian_courts_cases",
        "rishiai/indian-court-judgements-and-its-summaries",
        "maheshCoder/indian_court_cases"
    ]
    
    questions, answers, info = [], [], []

    for name in datasets:
        q, a, source = load_dataset_specific(name)
        if q and a:
            questions.extend(q)
            answers.extend(a)
            info.append({"name": source, "count": len(q)})
    
    # Create embeddings and index
    st.info("🧠 Creating embeddings and building index...")
    embedder = SentenceTransformer("law-ai/InLegalBERT")
    embeddings = embedder.encode(questions, show_progress_bar=True)
    
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings, dtype="float32"))

    # Save files for future use
    st.info("💾 Saving datasets for future use...")
    with open(questions_path, "wb") as f:
        pickle.dump(questions, f)
    with open(answers_path, "wb") as f:
        pickle.dump(answers, f)
    faiss.write_index(index, index_path)

    st.success("✅ Successfully loaded and cached datasets")
    return questions, answers, embedder, index, info

def load_dataset_specific(name):
    try:
        dataset = load_dataset(name, split="train")
    except Exception as e:
        print(f"Failed to load dataset {name}: {e}")
        return [], [], name

    cols = dataset.column_names
    questions, answers = [], []

    if name == "santoshtyss/indian_courts_cases" and "text" in cols:
        for entry in dataset:
            try:
                facts, judgment = extract_case_parts(entry["text"])
                questions.append(facts)
                answers.append(judgment)
            except Exception as e:
                print(f"Skipping entry due to parsing error: {e}")
                continue

    elif name == "rishiai/indian-court-judgements-and-its-summaries":
        for entry in dataset:
            summary = str(entry.get("Summary", "")).strip()
            judgment = str(entry.get("Judgment", "")).strip()
            if summary and judgment and len(summary) > 30 and len(judgment) > 100:
                questions.append(summary)
                answers.append(judgment)

    elif name == "maheshCoder/indian_court_cases":
        for entry in dataset:
            desc = entry.get("Case Description", "").strip()
            outcome = entry.get("Case Outcome", "").strip()
            if desc and outcome:
                questions.append(desc)
                answers.append(outcome)

    return questions, answers, name
