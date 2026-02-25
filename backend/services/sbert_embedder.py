"""
Service for embedding documents using SBERT
"""

from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL
import numpy as np

def embed_job_descriptions(job_descriptions: list[str]) -> np.ndarray:
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(job_descriptions, show_progress_bar=True, normalize_embeddings=True)
    return np.array(embeddings)

