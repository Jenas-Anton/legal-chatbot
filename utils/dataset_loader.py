from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
import streamlit as st
import numpy as np
from utils.extractors import extract_case_parts
import re

@st.cache_resource
def load_combined_datasets():
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
    
    # ✅ Use InLegalBERT for better legal-domain embeddings
    embedder = SentenceTransformer("law-ai/InLegalBERT")
    embeddings = embedder.encode(questions, show_progress_bar=True)
    
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings, dtype="float32"))

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
