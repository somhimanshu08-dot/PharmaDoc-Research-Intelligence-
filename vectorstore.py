"""
Local vector store using Chroma + Gemini's embedding API (no torch/sentence-
transformers needed -- keeps the app lightweight enough for free-tier hosting).
Stores chunk text plus metadata (doc name, page, drug name) so retrieval
results can be traced back to an exact source for citations.
"""

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
import google.generativeai as genai
from config import CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL_NAME, GOOGLE_API_KEY
from typing import List, Dict, Optional
import uuid

genai.configure(api_key=GOOGLE_API_KEY)


class GeminiEmbeddingFunction(EmbeddingFunction):
    """Calls Gemini's embedding API instead of running a local torch model."""
    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            result = genai.embed_content(
                model=EMBED_MODEL_NAME,
                content=text,
                task_type="retrieval_document",
            )
            embeddings.append(result["embedding"])
        return embeddings


_embed_fn = GeminiEmbeddingFunction()

_client = chromadb.PersistentClient(path=CHROMA_DIR)
_collection = _client.get_or_create_collection(
    name=COLLECTION_NAME, embedding_function=_embed_fn
)


def add_document_chunks(chunks: List[Dict], doc_name: str, drug_name: Optional[str] = None,
                         upload_date: Optional[str] = None):
    ids, texts, metadatas = [], [], []
    for c in chunks:
        ids.append(str(uuid.uuid4()))
        texts.append(c["text"])
        metadatas.append({
            "doc_name": doc_name,
            "page_num": c["page_num"],
            "drug_name": drug_name or "unknown",
            "upload_date": upload_date or "unknown",
        })

    if texts:
        _collection.add(ids=ids, documents=texts, metadatas=metadatas)


def query_chunks(query: str, n_results: int = 6, doc_names: Optional[List[str]] = None,
                  drug_names: Optional[List[str]] = None) -> List[Dict]:
    where_filter = None
    conditions = []
    if doc_names:
        conditions.append({"doc_name": {"$in": doc_names}})
    if drug_names:
        conditions.append({"drug_name": {"$in": drug_names}})

    if len(conditions) == 1:
        where_filter = conditions[0]
    elif len(conditions) > 1:
        where_filter = {"$and": conditions}

    results = _collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_filter,
    )

    hits = []
    for text, meta in zip(results["documents"][0], results["metadatas"][0]):
        hits.append({"text": text, "metadata": meta})
    return hits


def list_uploaded_docs() -> List[str]:
    all_meta = _collection.get()["metadatas"]
    return sorted({m["doc_name"] for m in all_meta}) if all_meta else []


def delete_document(doc_name: str):
    _collection.delete(where={"doc_name": doc_name})


def clear_all_documents():
    global _collection
    _client.delete_collection(name=COLLECTION_NAME)
    _collection = _client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=_embed_fn
    )