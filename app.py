import streamlit as st
import tempfile, os, time
import pandas as pd

from pdf_processor import extract_pages_with_ocr_fallback, extract_tables, chunk_pages
from extractor import extract_structured_fields
from vectorstore import add_document_chunks, list_uploaded_docs, delete_document, clear_all_documents
from rag_qa import answer_question
from comparison import compare_drugs
from summarizer import summarize_document, summarize_across_documents
from config import CHUNK_SIZE, CHUNK_OVERLAP


def safe_unlink(path: str, retries: int = 5, delay: float = 0.3):
    """
    Deletes a temp file, retrying briefly if it's still locked by another
    process (common on Windows when Camelot/Ghostscript or a PDF library
    hasn't released its file handle yet). Never crashes the app if cleanup
    ultimately fails -- it's just a temp file.
    """
    for attempt in range(retries):
        try:
            os.unlink(path)
            return
        except PermissionError:
            time.sleep(delay)
    print(f"[safe_unlink] Warning: could not delete temp file {path} (still locked).")


st.set_page_config(page_title="Document Insight Extractor", layout="wide")
st.title("🧬 PharmaDoc & Research Intelligence")
st.caption("Upload clinical trial / regulatory PDFs → ask questions, compare drugs, extract tables, get grounded citations.")

# ---- Session state ----
if "extractions" not in st.session_state:
    st.session_state.extractions = {}
if "raw_texts" not in st.session_state:
    st.session_state.raw_texts = {}
if "tables" not in st.session_state:
    st.session_state.tables = {}

# Key counter used to force-reset the file uploader + tag input after each
# successful upload, so the previous file/tag doesn't linger in the widgets.
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# ---- Sidebar: upload ----
with st.sidebar:
    st.header("Upload documents")
    uploaded_files = st.file_uploader(
        "Upload PDF(s)", type=["pdf"], accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )
    drug_tag = st.text_input(
        "Tag these uploads with a drug name (optional)",
        key=f"drug_tag_{st.session_state.uploader_key}",
    )

    if uploaded_files and st.button("Process uploaded files"):
        for f in uploaded_files:
            if f.name in st.session_state.extractions:
                continue  # already processed

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name

            with st.spinner(f"Processing {f.name}..."):
                pages = extract_pages_with_ocr_fallback(tmp_path)
                full_text = "\n".join(p["text"] for p in pages)
                st.session_state.raw_texts[f.name] = full_text

                chunks = chunk_pages(pages, CHUNK_SIZE, CHUNK_OVERLAP)
                add_document_chunks(chunks, doc_name=f.name, drug_name=drug_tag or None)

                extraction = extract_structured_fields(full_text, source_document=f.name)
                st.session_state.extractions[f.name] = extraction

                tables = extract_tables(tmp_path)
                st.session_state.tables[f.name] = tables

            safe_unlink(tmp_path)

        st.success(f"Processed {len(uploaded_files)} file(s).")
        # Force the uploader + tag input to reset to empty on the next render,
        # so the previous file selection/tag doesn't linger for the next upload.
        st.session_state.uploader_key += 1
        st.rerun()

    st.divider()
    st.subheader("Uploaded documents")
    for name in list_uploaded_docs():
        col1, col2 = st.columns([4, 1])
        col1.write(f"• {name}")
        if col2.button("🗑️", key=f"delete_{name}", help=f"Remove {name}"):
            delete_document(name)
            st.session_state.extractions.pop(name, None)
            st.session_state.raw_texts.pop(name, None)
            st.session_state.tables.pop(name, None)
            st.rerun()

    if list_uploaded_docs():
        st.divider()
        if st.button("🗑️ Clear all documents", type="secondary"):
            clear_all_documents()
            st.session_state.extractions.clear()
            st.session_state.raw_texts.clear()
            st.session_state.tables.clear()
            st.rerun()

# ---- Main tabs ----
tab_qa, tab_extract, tab_compare, tab_summary, tab_tables = st.tabs(
    ["💬 Ask Questions", "📄 Structured Extraction", "⚖️ Compare Drugs", "📝 Summaries", "📊 Extracted Tables"]
)

# --- Tab 1: Q&A with citations ---
with tab_qa:
    st.subheader("Ask a natural-language question across your documents")
    doc_filter = st.multiselect("Limit to specific documents (optional)", list_uploaded_docs())
    question = st.text_input("Your question", placeholder="What were the primary endpoints across these trials?")

    if st.button("Ask") and question:
        with st.spinner("Retrieving and answering..."):
            result = answer_question(question, doc_names=doc_filter or None)
        st.markdown("### Answer")
        st.write(result["answer"])

        st.markdown("### Sources")
        for c in result["citations"]:
            with st.expander(f"[{c['ref']}] {c['doc_name']} — page {c['page_num']}"):
                st.write(c["snippet"] + "...")

# --- Tab 2: Structured extraction viewer ---
with tab_extract:
    st.subheader("Structured fields extracted per document")
    if not st.session_state.extractions:
        st.info("Upload and process documents first.")
    for name, extraction in st.session_state.extractions.items():
        with st.expander(name):
            st.json(extraction.model_dump())

# --- Tab 3: Drug comparison (supports 2 or more drugs) ---
with tab_compare:
    st.subheader("Compare two or more drugs across their uploaded trials")
    drug_names_available = sorted({
        e.drug_name for e in st.session_state.extractions.values() if e.drug_name
    })

    if not drug_names_available:
        st.info("Upload and tag documents with a drug name first.")
    else:
        selected_drugs = st.multiselect(
            "Select drugs to compare (2 or more)",
            drug_names_available,
        )

        if st.button("Compare") and selected_drugs:
            if len(selected_drugs) < 2:
                st.warning("Select at least two drugs.")
            else:
                with st.spinner(f"Comparing {len(selected_drugs)} drugs..."):
                    try:
                        df = compare_drugs(list(st.session_state.extractions.values()), selected_drugs)
                        st.dataframe(df, use_container_width=True)
                    except ValueError as e:
                        st.error(str(e))

# --- Tab 4: Summaries ---
with tab_summary:
    st.subheader("Summaries")
    mode = st.radio("Summary type", ["Single document", "Across all documents"])

    if mode == "Single document":
        doc_choice = st.selectbox("Choose a document", list(st.session_state.extractions.keys()))
        if doc_choice and st.button("Generate summary"):
            with st.spinner("Summarizing..."):
                summary = summarize_document(
                    st.session_state.extractions[doc_choice],
                    st.session_state.raw_texts[doc_choice],
                )
            st.write(summary)
    else:
        if st.session_state.extractions and st.button("Generate executive summary"):
            with st.spinner("Summarizing across documents..."):
                summary = summarize_across_documents(list(st.session_state.extractions.values()))
            st.write(summary)

# --- Tab 5: Extracted tables (native PDF tables) ---
with tab_tables:
    st.subheader("Tables extracted directly from PDFs")
    if not any(st.session_state.tables.values()):
        st.info("No tables detected yet, or documents not processed.")
    for name, tables in st.session_state.tables.items():
        if not tables:
            continue
        st.markdown(f"**{name}**")
        for t in tables:
            st.caption(f"Page {t['page_num']}, table #{t['table_index']}")
            st.dataframe(t["dataframe"], use_container_width=True)