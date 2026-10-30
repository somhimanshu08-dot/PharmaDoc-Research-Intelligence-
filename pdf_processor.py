"""
Handles turning a raw PDF into:
  1. Page-level text (with OCR fallback for scanned pages)
  2. Extracted tables (native tables in the PDF, e.g. adverse-event tables)
"""

import pdfplumber
from pdf2image import convert_from_path
import pytesseract
from typing import List, Dict
import camelot


MIN_CHARS_TO_SKIP_OCR = 20  # if a page has less real text than this, treat it as scanned


def extract_pages_with_ocr_fallback(pdf_path: str) -> List[Dict]:
    """
    Returns a list of {page_num, text} dicts.
    If a page's native text extraction is empty/near-empty, falls back to OCR.
    """
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""

            if len(text.strip()) < MIN_CHARS_TO_SKIP_OCR:
                # Likely a scanned page -> render to image and OCR it
                text = _ocr_page(pdf_path, i)

            pages.append({"page_num": i, "text": text.strip()})
    return pages


def _ocr_page(pdf_path: str, page_num: int) -> str:
    """Convert a single PDF page to an image and run Tesseract OCR on it."""
    images = convert_from_path(pdf_path, first_page=page_num, last_page=page_num, dpi=300)
    if not images:
        return ""
    return pytesseract.image_to_string(images[0]).strip()


def extract_tables(pdf_path: str) -> List[Dict]:
    """
    Extracts native tables embedded in the PDF (e.g. an adverse-event rate table),
    as opposed to tables we ask the LLM to generate from prose.
    Returns a list of {page_num, table_index, dataframe}.
    """
    results = []
    try:
        tables = camelot.read_pdf(pdf_path, pages="all", flavor="lattice")
        if len(tables) == 0:
            # Some PDFs don't have ruled lines -> try the "stream" flavor instead
            tables = camelot.read_pdf(pdf_path, pages="all", flavor="stream")

        for idx, t in enumerate(tables):
            results.append({
                "page_num": t.page,
                "table_index": idx,
                "dataframe": t.df,
            })
    except Exception as e:
        # Camelot can fail on some malformed PDFs -- don't crash the whole pipeline
        print(f"[extract_tables] Warning: table extraction failed for {pdf_path}: {e}")

    return results


def chunk_pages(pages: List[Dict], chunk_size: int, overlap: int) -> List[Dict]:
    """
    Splits page-level text into overlapping chunks for embedding/retrieval,
    while keeping track of which page each chunk came from (for citations).
    """
    chunks = []
    for page in pages:
        text = page["text"]
        if not text:
            continue
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            chunks.append({
                "text": chunk_text,
                "page_num": page["page_num"],
            })
            start += chunk_size - overlap
    return chunks
