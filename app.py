import streamlit as st
import os
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="Indian Legal Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f4e79;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .question-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f4e79;
        margin: 1rem 0;
    }
    .answer-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #0066cc;
        margin: 1rem 0;
    }
    .context-box {
        background-color: #f9f9f9;
        padding: 0.8rem;
        border-radius: 0.3rem;
        font-size: 0.9rem;
        border: 1px solid #ddd;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models_and_data(dataset_choice="comprehensive"):
    """Load dataset, embedder, and build FAISS index"""
    with st.spinner("Loading legal database and models... This may take a few minutes."):
        
        # Multiple dataset options for better coverage
        if dataset_choice == "comprehensive":
            try:
                # Try the comprehensive Supreme Court dataset first
                dataset = load_dataset("mendeley/IndicLegalQA", split="train")
                questions = dataset["question"] 
                answers = dataset["answer"]
                st.info("📚 Loaded IndicLegalQA Dataset: 10,000 Q&A pairs from Supreme Court judgments")
            except:
                try:
                    # Fallback to Kaggle dataset
                    dataset = load_dataset("akshatgupta7/llm-fine-tuning-dataset-of-indian-legal-texts", split="train")
                    questions = dataset["input"] if "input" in dataset.column_names else dataset["question"]
                    answers = dataset["output"] if "output" in dataset.column_names else dataset["answer"]
                    st.info("📚 Loaded Kaggle Indian Legal Dataset: IPC, CrPC & Constitution")
                except:
                    # Final fallback
                    dataset = load_dataset("Raghvender/IndianLawDataset", split="train")
                    questions = dataset["question"]
                    answers = dataset["answer"]
                    st.info("📚 Loaded Basic IndianLawDataset")
        
        elif dataset_choice == "supreme_court":
            try:
                # Load the comprehensive Supreme Court dataset
                dataset = load_dataset("mendeley/IndicLegalQA", split="train")
                questions = dataset["question"]
                answers = dataset["answer"]
                st.info("📚 Loaded IndicLegalQA: 10,000 Supreme Court Q&A pairs")
            except:
                st.error("Could not load Supreme Court dataset, falling back to basic dataset")
                dataset = load_dataset("Raghvender/IndianLawDataset", split="train")
                questions = dataset["question"]
                answers = dataset["answer"]
        
        elif dataset_choice == "criminal_civil":
            try:
                # Load Kaggle comprehensive dataset
                dataset = load_dataset("akshatgupta7/llm-fine-tuning-dataset-of-indian-legal-texts", split="train")
                questions = dataset["input"] if "input" in dataset.column_names else dataset["question"]
                answers = dataset["output"] if "output" in dataset.column_names else dataset["answer"]
                st.info("📚 Loaded Criminal & Civil Law Dataset: IPC, CrPC, Constitution")
            except:
                st.error("Could not load criminal/civil dataset, falling back to basic dataset")
                dataset = load_dataset("Raghvender/IndianLawDataset", split="train")
                questions = dataset["question"]
                answers = dataset["answer"]
        
        # Enhanced embedding models for better legal understanding
        try:
            embedder = SentenceTransformer("law-ai/InCaseLawBERT")  # Better for case law
            st.success("✅ Loaded InCaseLawBERT for enhanced legal embeddings")
        except:
            try:
                embedder = SentenceTransformer("law-ai/InLegalBERT")  # Fallback
                st.success("✅ Loaded InLegalBERT embeddings")
            except:
                embedder = SentenceTransformer("all-MiniLM-L6-v2")  # General fallback
                st.warning("⚠️ Using general embeddings (legal models unavailable)")
        
        # Build FAISS index
        question_embeddings = embedder.encode(questions, convert_to_numpy=True, show_progress_bar=False)
        dimension = question_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(question_embeddings)
        
        return questions, answers, embedder, index

def retrieve_similar_questions(query, embedder, index, questions, answers, k=3):
    """Retrieve similar questions using FAISS"""
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)
    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "question": questions[idx],
            "answer": answers[idx],
            "similarity_score": float(distances[0][i])
        })
    return results

def generate_answer_with_gemini(query, retrieved_qas, api_key):
    """Generate answer using Gemini"""
    try:
        genai.configure(api_key=api_key)
        
        context = "\n\n".join([f"Q: {qa['question']}\nA: {qa['answer']}" for qa in retrieved_qas])
        
        prompt = f"""You are an expert legal assistant specializing in Indian law. 

Based on the following retrieved legal Q&A context, provide a comprehensive and accurate answer to the user's question:

CONTEXT:
{context}

USER'S QUESTION: {query}

INSTRUCTIONS:
- Provide a clear, structured answer based on the context
- Reference relevant legal provisions, acts, or sections when applicable
- If the context doesn't fully address the question, mention this limitation
- Use professional legal language but keep it understandable
- Include practical implications when relevant

ANSWER:"""
        
        # Try different model names and API methods
        try:
            # Method 1: Try gemini-1.5-flash (newer model)
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            return response.text
        except:
            try:
                # Method 2: Try gemini-1.0-pro
                model = genai.GenerativeModel('gemini-1.0-pro')
                response = model.generate_content(prompt)
                return response.text
            except:
                try:
                    # Method 3: Use chat completion
                    response = genai.chat(
                        model="models/chat-bison-001",
                        messages=[{"content": prompt}],
                        temperature=0.2
                    )
                    return response.last
                except:
                    # Method 4: Use generate_text (legacy)
                    response = genai.generate_text(
                        model="models/text-bison-001",
                        prompt=prompt,
                        temperature=0.2,
                        max_output_tokens=512
                    )
                    return response.result if response.result else "No response generated"
                    
    except Exception as e:
        return f"Error generating response: {str(e)}. Please check your API key and try again."

def main():
    # Header
    st.markdown('<h1 class="main-header">⚖️ Indian Legal Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Get accurate answers to your Indian legal queries powered by AI</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # API Key input
        api_key = st.text_input("Google API Key", type="password", help="Enter your Google Gemini API key")
        
        # Add model testing button
        if api_key and st.button("🔍 Test API & List Models"):
            try:
                genai.configure(api_key=api_key)
                models = genai.list_models()
                st.success("✅ API key is valid!")
                with st.expander("Available Models"):
                    for model in models:
                        if 'generateContent' in model.supported_generation_methods:
                            st.write(f"✅ {model.name}")
                        else:
                            st.write(f"❌ {model.name} (not supported for generateContent)")
            except Exception as e:
                st.error(f"❌ API Error: {str(e)}")
        
        st.header("📊 Settings")
        
        # Dataset selection
        dataset_choice = st.selectbox(
            "Select Legal Database:",
            options=["comprehensive", "supreme_court", "criminal_civil"],
            format_func=lambda x: {
                "comprehensive": "🏛️ Comprehensive (Auto-select best available)",
                "supreme_court": "⚖️ Supreme Court Judgments (10K Q&A)",
                "criminal_civil": "📜 Criminal & Civil Law (IPC/CrPC/Constitution)"
            }[x],
            help="Choose the legal database that best fits your needs"
        )
        
        num_similar = st.slider("Number of similar cases to retrieve", 1, 5, 3)
        
        st.header("💡 Dataset Information")
        
        if dataset_choice == "comprehensive":
            st.info("""
            **Auto-selects the best available dataset:**
            1. **IndicLegalQA** (10,000 Supreme Court Q&A pairs)
            2. **Kaggle Legal Dataset** (IPC, CrPC, Constitution)
            3. **Basic Dataset** (fallback)
            """)
        elif dataset_choice == "supreme_court":
            st.info("""
            **IndicLegalQA Dataset:**
            - 10,000 question-answer pairs
            - Derived from 1,256 Supreme Court judgments
            - Covers criminal (538) and civil (718) cases
            - Constitutional, property, criminal law topics
            """)
        else:
            st.info("""
            **Criminal & Civil Law Dataset:**
            - Indian Penal Code (IPC) Q&A
            - Criminal Procedure Code (CrPC) Q&A  
            - Indian Constitution provisions
            - Comprehensive legal act coverage
            """)
        
        st.header("ℹ️ About")
        st.info("""
        This legal assistant uses:
        - **InLegalBERT**: Specialized legal embeddings
        - **FAISS**: Fast similarity search
        - **Gemini Pro**: Advanced answer generation
        - **Indian Law Dataset**: Comprehensive legal Q&A database
        """)
        
        st.warning("⚠️ **Disclaimer**: This is for informational purposes only. Always consult a qualified lawyer for legal advice.")
    
    # Load models and data
    try:
        questions, answers, embedder, index = load_models_and_data()
        st.success("✅ Legal database loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Ask Your Legal Question")
        query = st.text_area(
            "Enter your legal question here:",
            height=100,
            placeholder="e.g., What are the grounds for divorce under Hindu Marriage Act?",
            help="Ask any question related to Indian law"
        )
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            search_button = st.button("🔍 Get Answer", type="primary", use_container_width=True)
        with col_btn2:
            clear_button = st.button("🗑️ Clear", use_container_width=True)
    
    with col2:
        st.subheader("📋 Quick Examples")
        examples = [
            "Rights of tenants under Rent Control Act",
            "Procedure for company registration",
            "Grounds for anticipatory bail",
            "Property rights of women",
            "Consumer protection rights"
        ]
        
        for example in examples:
            if st.button(f"💡 {example}", key=example, use_container_width=True):
                st.session_state.example_query = example
    
    # Handle example selection
    if hasattr(st.session_state, 'example_query'):
        query = st.session_state.example_query
        delattr(st.session_state, 'example_query')
        st.rerun()
    
    # Clear functionality
    if clear_button:
        st.rerun()
    
    # Process query
    if search_button and query and api_key:
        with st.spinner("🔍 Searching legal database and generating answer..."):
            # Retrieve similar questions
            retrieved_qas = retrieve_similar_questions(
                query, embedder, index, questions, answers, k=num_similar
            )
            
            # Generate answer
            answer = generate_answer_with_gemini(query, retrieved_qas, api_key)
            
            # Display results
            st.markdown("---")
            st.subheader("📝 Generated Answer")
            
            # Main answer
            st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)
            
            # Similar cases section
            with st.expander("📚 Similar Cases from Database", expanded=False):
                for i, qa in enumerate(retrieved_qas, 1):
                    st.markdown(f"**Case {i}:**")
                    st.markdown(f'<div class="context-box"><strong>Q:</strong> {qa["question"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="context-box"><strong>A:</strong> {qa["answer"]}</div>', unsafe_allow_html=True)
                    st.markdown(f"*Similarity Score: {qa['similarity_score']:.4f}*")
                    st.markdown("---")
            
            # Metadata
            with st.expander("🔍 Query Information"):
                st.write(f"**Query processed at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                st.write(f"**Similar cases retrieved:** {len(retrieved_qas)}")
                st.write(f"**Database size:** {len(questions)} legal Q&A pairs")
    
    elif search_button and not api_key:
        st.error("❌ Please enter your Google API key in the sidebar")
    elif search_button and not query:
        st.error("❌ Please enter a legal question")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        <p>🏛️ <strong>Indian Legal Assistant</strong> | Powered by AI | For Educational Purposes Only</p>
        <p>Always consult with a qualified legal professional for official legal advice</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()