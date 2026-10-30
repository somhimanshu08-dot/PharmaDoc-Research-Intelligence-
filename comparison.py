"""
Compares TWO OR MORE drugs by grouping all structured extractions that
belong to each drug_name, then asking Gemini to align them into one
comparison table (rather than comparing raw documents 1:1).
"""

import json
import pandas as pd
import google.generativeai as genai
from config import GOOGLE_API_KEY, MODEL_NAME
from schema import TrialExtraction
from typing import List

genai.configure(api_key=GOOGLE_API_KEY)
_model = genai.GenerativeModel(MODEL_NAME)

_COMPARISON_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "drug_name": {"type": "string"},
            "phase": {"type": "string"},
            "indication": {"type": "string"},
            "primary_endpoint": {"type": "string"},
            "sample_size": {"type": "string"},
            "adverse_events_summary": {"type": "string"},
            "conclusion": {"type": "string"},
        },
        "required": ["drug_name"],
    },
}


def compare_drugs(extractions: List[TrialExtraction], drug_names: List[str]) -> pd.DataFrame:
    """
    extractions: all structured extractions collected so far (one per uploaded doc)
    drug_names: 2 or more drug names to compare (case-insensitive match against
                each extraction's drug_name field)

    Groups extractions by each requested drug, then asks the model to produce
    ONE row per drug -- merging data across multiple trials for the same drug
    if present -- so the output scales to any number of drugs, not just two.
    """
    if len(drug_names) < 2:
        raise ValueError("Select at least two drugs to compare.")

    grouped = {}
    missing = []
    for drug in drug_names:
        matches = [e.model_dump() for e in extractions if (e.drug_name or "").lower() == drug.lower()]
        if not matches:
            missing.append(drug)
        else:
            grouped[drug] = matches

    if missing:
        raise ValueError(f"Could not find extracted data for: {', '.join(missing)}.")

    data_blocks = "\n\n".join(
        f"DRUG {i+1} ({drug}) DATA:\n{json.dumps(docs, indent=2)}"
        for i, (drug, docs) in enumerate(grouped.items())
    )

    prompt = (
        f"You are given structured extraction data (JSON) for {len(grouped)} drugs, "
        "each possibly from multiple trials. Merge each drug's data into ONE row per "
        "drug (there should be exactly one row per drug in your output). "
        "For adverse_events_summary, summarize the most notable adverse events across "
        "that drug's trials in a short phrase.\n\n"
        f"{data_blocks}"
    )

    response = _model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=_COMPARISON_SCHEMA,
            max_output_tokens=1800,
        ),
    )

    rows = json.loads(response.text)
    return pd.DataFrame(rows)