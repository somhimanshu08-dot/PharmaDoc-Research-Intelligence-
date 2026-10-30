import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = "gemini-3.1-flash-lite"                   # generation model
EMBED_MODEL_NAME = "models/gemini-embedding-001"    # Gemini embedding model (API-based, no torch needed)

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "trial_documents"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

if not GOOGLE_API_KEY:
    raise EnvironmentError(
        "GOOGLE_API_KEY not set. Copy .env.example to .env and add your key."
    )