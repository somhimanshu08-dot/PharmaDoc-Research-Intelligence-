"""
Forces Gemini to return structured JSON matching TrialExtraction schema,
using Gemini's controlled generation (response_schema + response_mime_type),
so we get validated, predictable output rather than parsing free text.
"""

import json
import google.generativeai as genai
from config import GOOGLE_API_KEY, MODEL_NAME
from schema import GEMINI_EXTRACTION_SCHEMA, TrialExtraction

genai.configure(api_key=GOOGLE_API_KEY)
_model = genai.GenerativeModel(MODEL_NAME)


def extract_structured_fields(full_text: str, source_document: str) -> TrialExtraction:
    """
    Sends the full document text (or a large representative excerpt) to Gemini
    and forces JSON output matching GEMINI_EXTRACTION_SCHEMA.
    """
    # Guard against extremely long documents -- keep prompt within a safe budget.
    excerpt = full_text[:15000]

    prompt = (
        "Extract the key structured fields from this clinical/regulatory "
        "document. If a field is not present in the text, omit it rather "
        "than guessing.\n\n"
        f"DOCUMENT TEXT:\n{excerpt}"
    )

    response = _model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=GEMINI_EXTRACTION_SCHEMA,
            max_output_tokens=1500,
        ),
    )

    data = json.loads(response.text)
    data["source_document"] = source_document

    return TrialExtraction(**data)
