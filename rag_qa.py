"""
Natural-language Q&A over uploaded documents. Retrieves relevant chunks,
asks Gemini to answer using ONLY those chunks, and requires it to tag every
claim with a [source] marker we map back to (doc_name, page_num) for citations.
"""

import google.generativeai as genai
from config import GOOGLE_API_KEY, MODEL_NAME
from vectorstore import query_chunks
from typing import List, Dict, Optional

genai.configure(api_key=GOOGLE_API_KEY)
_model = genai.GenerativeModel(MODEL_NAME)


def answer_question(question: str, doc_names: Optional[List[str]] = None,
                     drug_names: Optional[List[str]] = None) -> Dict:
    hits = query_chunks(question, n_results=6, doc_names=doc_names, drug_names=drug_names)

    if not hits:
        return {"answer": "No relevant content found in the uploaded documents.", "citations": []}

    # Build a numbered context block so the model can reference [1], [2], etc.
    context_blocks = []
    for i, h in enumerate(hits, start=1):
        meta = h["metadata"]
        context_blocks.append(
            f"[{i}] (Source: {meta['doc_name']}, page {meta['page_num']})\n{h['text']}"
        )
    context_text = "\n\n".join(context_blocks)

    prompt = (
        "Answer the question using ONLY the numbered source excerpts below. "
        "After every claim, cite the excerpt number(s) it came from, like [1] or [1,3]. "
        "If the excerpts don't contain enough information, say so explicitly rather "
        "than guessing.\n\n"
        f"SOURCE EXCERPTS:\n{context_text}\n\n"
        f"QUESTION: {question}"
    )

    response = _model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(max_output_tokens=800),
    )

    answer_text = response.text

    # Map excerpt numbers back to real citations for the UI
    citations = [
        {
            "ref": i,
            "doc_name": h["metadata"]["doc_name"],
            "page_num": h["metadata"]["page_num"],
            "snippet": h["text"][:250],
        }
        for i, h in enumerate(hits, start=1)
    ]

    return {"answer": answer_text, "citations": citations}
