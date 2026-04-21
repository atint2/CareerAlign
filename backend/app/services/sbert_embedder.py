"""
Service for embedding documents using SBERT
Includes singleton loader
"""

from sentence_transformers import SentenceTransformer
from backend.app.config import EMBEDDING_MODEL
import numpy as np

class SBERTEmbeddingService:
    """
    SBERT embedding service for resumes and job descriptions.
    """

    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of texts using SBERT.
        """
        embeddings = self.model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        return np.array(embeddings)

# Singleton pattern to ensure only one instance of the embedding service is created
_instance = None

def get_sbert_service() -> SBERTEmbeddingService:
    global _instance
    if _instance is None:
        _instance = SBERTEmbeddingService()
    return _instance