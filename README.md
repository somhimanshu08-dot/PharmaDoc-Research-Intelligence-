# PharmaDoc & Research Intelligence

A GenAI web app for extracting structured insights from clinical trial / regulatory
PDFs and Research Papers

## Features
- **Multi-PDF upload** with drug-name tagging
- **Natural-language Q&A** over documents (RAG)
- **Structured JSON extraction** drug, phase, indication, endpoint, adverse events, conclusion
- **Drug-vs-drug comparison** tables, aggregated across multiple trials per drug
- **Summaries** — per-document or executive summary across all uploads
- **Native table extraction** from PDFs via Camelot
- **OCR fallback** for scanned pages (Tesseract)
- **Citation grounding** — every Q&A answer links back to (document, page number)

## Architecture
```
PharmaDoc-Research-Intelligence/
│
├── app.py
├── config.py
├── schema.py
├── extractor.py
├── pdf_processor.py
├── vectorstore.py
├── rag_qa.py
├── comparison.py
├── summarizer.py
├── requirements.txt
├── .env.example
└── README.md
```
## Live Demo

[Click here to try PharmaDoc & Research Intelligence](https://pharmadoc-research-intelligence-e4pggltxh3rdwmfgomomwp.streamlit.app/)
## Setup

1. **Install system dependencies** (for OCR and table extraction):
   ```bash
   # macOS
   brew install tesseract poppler ghostscript

   # Ubuntu/Debian
   sudo apt-get install tesseract-ocr poppler-utils ghostscript python3-tk
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set your API key:**
   ```bash
   cp .env.example .env
   # then edit .env and add your GOOGLE_API_KEY (get one at aistudio.google.com/apikey)
   ```

4. **Run the app:**
   ```bash
   streamlit run app.py
   ```

## Usage
1. Upload one or more PDFs in the sidebar, optionally tagging each batch with a drug name.
2. Click **Process uploaded files** — this extracts text, tables, OCRs scanned pages,
   runs structured extraction, and indexes chunks for retrieval.
3. Use the tabs to ask questions, view structured data, compare multiple drugs, generate
   summaries, or inspect extracted tables.

