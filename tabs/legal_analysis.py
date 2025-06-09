import streamlit as st
import time
from utils.embedding_search import search_similar_cases
from utils.gemini_interface import generate_case_summary, generate_answer

def show_legal_analysis_tab(context):
    """Legal Question Analysis Tab"""
    st.header("💬 Legal Question Analysis")
    st.markdown("Ask legal questions and get analysis based on similar cases from our database.")
    
    if not context['datasets_loaded']:
        st.warning("⚠️ Dataset loading failed. This feature requires legal case databases.")
        return
    
    query = st.text_area(
        "Enter your legal question or scenario", 
        height=150,
        placeholder="e.g., What are the implications of breach of contract in commercial agreements?"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        analyze_btn = st.button("🔍 Analyze Question", type="primary")
    
    if analyze_btn:
        if not query:
            st.error("Please enter a legal question.")
        else:
            with st.spinner("Searching similar cases and generating analysis..."):
                try:
                    similar_cases = search_similar_cases(
                        query, 
                        context['index'], 
                        context['questions'], 
                        context['answers'], 
                        context['num_cases']
                    )
                    
                    # Generate case summaries
                    summarized_cases = []
                    progress_bar = st.progress(0)
                    for i, case in enumerate(similar_cases):
                        summary = generate_case_summary(
                            case['question'], 
                            case['answer'], 
                            context['api_key']
                        )
                        summarized_cases.append({
                            "summary": summary,
                            "similarity": case['similarity']
                        })
                        progress_bar.progress((i + 1) / len(similar_cases))
                    
                    # Generate comprehensive analysis
                    analysis = generate_answer(query, similar_cases, context['api_key'])
                    
                    # Display results
                    st.subheader("📋 Legal Analysis")
                    st.markdown(analysis)
                    
                    # Download analysis
                    st.download_button(
                        label="💾 Download Analysis",
                        data=f"Query: {query}\n\n{analysis}",
                        file_name=f"legal_analysis_{int(time.time())}.txt",
                        mime="text/plain"
                    )
                    
                    st.subheader("📚 Similar Cases (Summarized)")
                    for i, case in enumerate(summarized_cases, 1):
                        with st.expander(f"Case {i} - Similarity: {case['similarity']:.1f}%"):
                            st.markdown(case["summary"])
                            
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    st.exception(e)