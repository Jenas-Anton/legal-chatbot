import streamlit as st
import tempfile
import os
from utils.embedding_search import search_similar_cases
from utils.gemini_interface import generate_case_summary

def show_case_prediction_tab(context):
    """Case Outcome Prediction Tab"""
    st.header("📄 Case Outcome Prediction")
    st.markdown("Upload case documents to predict outcomes based on similar historical cases.")
    
    if not context['datasets_loaded']:
        st.warning("⚠️ Dataset loading failed. This feature requires legal case databases.")
        return
    
    uploaded_case_file = st.file_uploader(
        "Upload Case Document", 
        type=['pdf', 'docx', 'doc'], 
        key="case_upload",
        help="Upload PDF, DOCX, or DOC files containing case details"
    )
    
    if uploaded_case_file:
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.' + uploaded_case_file.name.split('.')[-1]) as tmp_file:
                tmp_file.write(uploaded_case_file.getvalue())
                tmp_path = tmp_file.name

            # Extract text based on file type
            with st.spinner("Extracting text from document..."):
                if uploaded_case_file.name.endswith('.pdf'):
                    document_text = context['case_predictor'].extract_text_from_pdf(tmp_path)
                else:
                    document_text = context['case_predictor'].extract_text_from_doc(tmp_path)

            # Clean up temp file
            os.unlink(tmp_path)

            # Show extracted text preview
            with st.expander("📄 Extracted Text Preview"):
                st.text_area(
                    "Document Content", 
                    document_text[:1000] + "..." if len(document_text) > 1000 else document_text, 
                    height=200
                )

            # Analyze document
            with st.spinner("Analyzing document and searching similar cases..."):
                similar_cases = search_similar_cases(
                    document_text[:1000], 
                    context['index'], 
                    context['questions'], 
                    context['answers'], 
                    context['num_cases']
                )
                
                summarized_cases = []
                for case in similar_cases:
                    summary = generate_case_summary(
                        case['question'], 
                        case['answer'], 
                        context['api_key']
                    )
                    summarized_cases.append({
                        "summary": summary, 
                        "similarity": case['similarity']
                    })
                
                prediction_result = context['case_predictor'].predict_case(document_text, summarized_cases)

            # Display prediction result
            confidence = prediction_result['confidence']
            color = "green" if confidence > 0.7 else "orange" if confidence > 0.5 else "red"
            
            st.markdown(f"""
                <div style="padding: 1.5rem; border-radius: 0.5rem; border-left: 5px solid {color}; background-color: {color}15; margin: 1rem 0;">
                    <h3 style="color: {color}; margin: 0;">🎯 Prediction: {prediction_result['prediction']}</h3>
                    <p style="margin: 0.5rem 0 0 0; font-size: 1.1em;">Confidence: {confidence:}</p>
                </div>
            """, unsafe_allow_html=True)

            # Display similar cases
            st.subheader("📚 Most Similar Historical Cases")
            for i, case in enumerate(summarized_cases, 1):
                with st.expander(f"📋 Case {i} - Similarity: {case['similarity']:.1f}%"):
                    st.markdown(case["summary"])

        except Exception as e:
            st.error(f"❌ Error processing document: {str(e)}")
            st.exception(e)