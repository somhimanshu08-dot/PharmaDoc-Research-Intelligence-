from pydantic import BaseModel, Field
from typing import List, Optional


class AdverseEvent(BaseModel):
    event: str
    frequency: Optional[str] = None   # e.g. "12%", "3/50 patients"
    severity: Optional[str] = None    # e.g. "mild", "serious"


class TrialExtraction(BaseModel):
    """Structured fields we force the LLM to fill from a trial/regulatory document."""
    drug_name: Optional[str] = None
    phase: Optional[str] = None
    indication: Optional[str] = None
    primary_endpoint: Optional[str] = None
    sample_size: Optional[str] = None
    adverse_events: List[AdverseEvent] = Field(default_factory=list)
    conclusion: Optional[str] = None
    source_document: Optional[str] = None   # filled in by our code, not the LLM


# JSON schema for Gemini's controlled generation (response_schema param).
# Gemini uses a restricted OpenAPI-style schema -- notably no "required" combined
# with optional fields the same way Claude's tool schema does, so this is kept
# separately rather than reused.
GEMINI_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "drug_name": {"type": "string"},
        "phase": {"type": "string", "description": "e.g. Phase 1, Phase 2, Phase 3"},
        "indication": {"type": "string"},
        "primary_endpoint": {"type": "string"},
        "sample_size": {"type": "string"},
        "adverse_events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "event": {"type": "string"},
                    "frequency": {"type": "string"},
                    "severity": {"type": "string"},
                },
                "required": ["event"],
            },
        },
        "conclusion": {"type": "string"},
    },
    "required": ["drug_name", "conclusion"],
}


# JSON schema for Gemini's controlled generation (response_schema), which forces
# the model to return JSON matching this shape -- Gemini's equivalent of Claude's
# tool-use pattern, but simpler: no tool_choice/tool_use block parsing needed.
GEMINI_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "drug_name": {"type": "string"},
        "phase": {"type": "string"},
        "indication": {"type": "string"},
        "primary_endpoint": {"type": "string"},
        "sample_size": {"type": "string"},
        "adverse_events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "event": {"type": "string"},
                    "frequency": {"type": "string"},
                    "severity": {"type": "string"},
                },
                "required": ["event"],
            },
        },
        "conclusion": {"type": "string"},
    },
    "required": ["drug_name", "conclusion"],
}

# JSON schema handed to Claude as a tool definition, so the model MUST return
# data matching this shape (function calling / tool-use pattern).
EXTRACTION_TOOL = {
    "name": "record_trial_extraction",
    "description": "Record structured fields extracted from a clinical trial or regulatory document.",
    "input_schema": {
        "type": "object",
        "properties": {
            "drug_name": {"type": "string"},
            "phase": {"type": "string", "description": "e.g. Phase 1, Phase 2, Phase 3"},
            "indication": {"type": "string"},
            "primary_endpoint": {"type": "string"},
            "sample_size": {"type": "string"},
            "adverse_events": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "event": {"type": "string"},
                        "frequency": {"type": "string"},
                        "severity": {"type": "string"},
                    },
                    "required": ["event"],
                },
            },
            "conclusion": {"type": "string"},
        },
        "required": ["drug_name", "conclusion"],
    },
}
