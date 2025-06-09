import streamlit as st
import re
import time
from document_generator import (
    DOCUMENT_DESCRIPTIONS, DOCUMENT_TEMPLATES, PLACEHOLDER_EXAMPLES,
    load_template, extract_placeholders, validate_inputs,
    generate_clause, export_to_docx, export_to_pdf
)

def show_document_generator_tab():
    """Legal Document Generator Tab"""
    st.header("📃 AI-Powered Legal Document Generator")
    st.markdown("Generate professional legal documents with AI-powered clause creation.")

    selected_doc = st.selectbox(
        "Select Agreement Type", 
        list(DOCUMENT_TEMPLATES.keys()),
        help="Choose the type of legal document you want to generate"
    )

    st.info(f"📋 **About {selected_doc}:** {DOCUMENT_DESCRIPTIONS[selected_doc]}")

    template = load_template(DOCUMENT_TEMPLATES[selected_doc])
    placeholders = extract_placeholders(template)

    manual_inputs = {}
    ai_inputs = []

    with st.form("draft_form"):
        st.subheader("📝 Fill Required Information")

        col1, col2 = st.columns(2)
        manual_placeholders = [ph for ph in placeholders if not ph.startswith("clause_")]

        for i, ph in enumerate(manual_placeholders):
            target_col = col1 if i % 2 == 0 else col2
            with target_col:
                label = f"{ph.replace('_', ' ').title()}"
                help_text = PLACEHOLDER_EXAMPLES.get(ph, None)
                manual_inputs[ph] = st.text_input(
                    label,
                    key=f"manual_{ph}",
                    help=help_text,
                    placeholder=""  # No visible placeholder text
                )

        ai_inputs = [ph for ph in placeholders if ph.startswith("clause_")]

        if ai_inputs:
            st.subheader("🤖 AI-Generated Clauses")
            st.info(
                f"The following clause(s) will be generated: "
                f"{', '.join([clause.replace('clause_', '').replace('_', ' ').title() for clause in ai_inputs])}"
            )

        col1, col2 = st.columns([1, 1])
        with col1:
            export_format = st.selectbox("Export Format", ["PDF", "DOCX"])
        with col2:
            submitted = st.form_submit_button("🚀 Generate Legal Document", type="primary")

    if submitted:
        if validate_inputs(manual_inputs):
            st.success("✅ All fields validated.")
            with st.spinner("Generating AI clauses..."):
                progress_bar = st.progress(0)

                for i, ai_ph in enumerate(ai_inputs):
                    clause_name = ai_ph.replace("clause_", "").replace("_", " ").title()
                    prompt = (
                        f"You are an expert legal assistant. Generate a comprehensive, formal legal clause "
                        f"for the topic: '{clause_name}' in the context of a {selected_doc}."
                    )

                    try:
                        ai_text = generate_clause(prompt)
                        manual_inputs[ai_ph] = ai_text
                        with st.expander(f"📄 {clause_name}"):
                            st.write(ai_text)
                    except Exception as e:
                        manual_inputs[ai_ph] = f"[Error generating {clause_name}: {e}]"
                        st.error(f"Error generating {clause_name}: {e}")

                    progress_bar.progress((i + 1) / len(ai_inputs))

            final_text = template
            for k, v in manual_inputs.items():
                pattern = re.escape(k)
                final_text = re.sub(r"{{\s*" + pattern + r"\s*}}", str(v), final_text)

            st.subheader("📃 Preview")
            st.text_area("Document Preview", final_text, height=400, key="preview", disabled=True)

            col1, col2, _ = st.columns([1, 1, 2])
            try:
                if export_format == "DOCX":
                    with col1:
                        docx_data = export_to_docx(final_text)
                        st.download_button(
                            label="📄 Download DOCX",
                            data=docx_data.getvalue(),
                            file_name=f"{selected_doc.replace(' ', '_')}_{int(time.time())}.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )
                else:
                    with col1:
                        pdf_data = export_to_pdf(final_text)
                        st.download_button(
                            label="📄 Download PDF",
                            data=pdf_data.getvalue(),
                            file_name=f"{selected_doc.replace(' ', '_')}_{int(time.time())}.pdf",
                            mime="application/pdf"
                        )
                with col2:
                    st.download_button(
                        label="📝 Download TXT",
                        data=final_text,
                        file_name=f"{selected_doc.replace(' ', '_')}_{int(time.time())}.txt",
                        mime="text/plain"
                    )
            except Exception as e:
                st.error(f"Export error: {e}")
        else:
            st.warning("⚠️ Fill all required fields first.")
