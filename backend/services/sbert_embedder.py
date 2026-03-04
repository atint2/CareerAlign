"""
Service for embedding documents using SBERT
"""

from sentence_transformers import SentenceTransformer
from backend.config import EMBEDDING_MODEL
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

