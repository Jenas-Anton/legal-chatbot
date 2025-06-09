import streamlit as st
from pathlib import Path
from app.utils import save_uploaded_file

def show_document_summary_tab(context):
    """Document Summarization & Q&A Tab"""
    st.header("📋 Document Summarization & Q&A")
    st.markdown("Upload legal documents for AI-powered summarization and custom queries.")
    
    uploaded_doc = st.file_uploader(
        "Upload Legal Document", 
        type=['pdf', 'docx', 'txt'], 
        key="summary_upload",
        help="Upload PDF, DOCX, or TXT files for analysis"
    )

    if uploaded_doc:
        file_path = save_uploaded_file(uploaded_doc)
        st.success(f"✅ Uploaded: **{uploaded_doc.name}** ({len(uploaded_doc.getvalue()):,} bytes)")

        summary_tab, query_tab = st.tabs(["📋 Document Summarization", "❓ Custom Q&A"])

        with summary_tab:
            st.subheader("📄 Generate Document Summary")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                summary_type = st.selectbox(
                    "Select Summary Type",
                    options=[
                        ("executive", "Executive Summary"),
                        ("detailed", "Detailed Summary"),
                        ("key_points", "Key Points"),
                        ("roles_parties", "Roles & Parties"),
                        ("timeline", "Timeline"),
                        ("risk_analysis", "Risk Analysis"),
                        ("comprehensive", "Comprehensive Analysis")
                    ],
                    format_func=lambda x: x[1]
                )
            
            with col2:
                generate_summary_btn = st.button("🔍 Generate Summary", type="primary")

            if generate_summary_btn:
                with st.spinner(f"Generating {summary_type[1].lower()}..."):
                    try:
                        summary = context['assistant'].summarize_document(file_path, summary_type[0], None)
                        
                        st.subheader(f"📄 {summary_type[1]}")
                        st.markdown(summary)
                        
                        # Download option
                        st.download_button(
                            label="💾 Download Summary",
                            data=summary,
                            file_name=f"summary_{Path(uploaded_doc.name).stem}_{summary_type[0]}.txt",
                            mime="text/plain"
                        )
                    except Exception as e:
                        st.error("❌ Error generating summary.")
                        st.exception(e)

        with query_tab:
            st.subheader("❓ Ask Questions About the Document")
            
            prompt = st.text_area(
                "Enter your question", 
                placeholder="e.g., What are the key payment terms mentioned in this contract?", 
                height=100
            )

            if st.button("🔍 Get Answer", type="primary"):
                if not prompt.strip():
                    st.error("Please enter a question.")
                else:
                    with st.spinner("Processing your query..."):
                        try:
                            answer = context['assistant'].summarize_document(file_path, "custom", prompt)
                            
                            st.subheader("💡 Answer")
                            st.markdown(answer)
                            
                            # Download option
                            st.download_button(
                                label="💾 Download Q&A",
                                data=f"**Question:** {prompt}\n\n**Answer:** {answer}",
                                file_name=f"qa_result_{Path(uploaded_doc.name).stem}.txt",
                                mime="text/plain"
                            )
                        except Exception as e:
                            st.error("❌ Error processing query.")
                            st.exception(e)