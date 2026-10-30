"""
Generates plain-language summaries -- either for a single document
(using its structured extraction + raw text) or across all uploaded documents.
"""

import google.generativeai as genai
from config import GOOGLE_API_KEY, MODEL_NAME
from schema import TrialExtraction
from typing import List

genai.configure(api_key=GOOGLE_API_KEY)
_model = genai.GenerativeModel(MODEL_NAME)


def summarize_document(extraction: TrialExtraction, raw_text_excerpt: str) -> str:
    prompt = (
        "Write a concise 4-6 sentence plain-language summary of this clinical/regulatory "
        "document for a pharma strategy team. Use the structured data as your factual "
        "anchor and the raw excerpt for extra context. Avoid jargon where possible.\n\n"
        f"STRUCTURED DATA:\n{extraction.model_dump_json(indent=2)}\n\n"
        f"RAW EXCERPT:\n{raw_text_excerpt[:3000]}"
    )
    response = _model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(max_output_tokens=400),
    )
    return response.text


def summarize_across_documents(extractions: List[TrialExtraction]) -> str:
    combined = "\n\n".join(e.model_dump_json(indent=2) for e in extractions)
    prompt = (
        "Write a short executive summary (6-8 sentences) synthesizing the key findings "
        "across ALL of these documents -- highlight any notable patterns, differences "
        "in efficacy/safety across drugs, and overall takeaways for a pharma team.\n\n"
        f"{combined}"
    )
    response = _model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(max_output_tokens=500),
    )
    return response.text
