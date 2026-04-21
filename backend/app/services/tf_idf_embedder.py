"""
Service for embedding documents using TF-IDF
Includes singleton loader and keyword utility functions
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from backend.app.config import CUSTOM_STOPWORDS
import pickle
import numpy as np

class TFIDFEmbeddingService:
    """
    TF-IDF embedding service for resumes and job descriptions.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words=CUSTOM_STOPWORDS,
            max_df=0.8,
            min_df=5,
            ngram_range=(1, 2),
            max_features=5000
        )

    def fit(self, texts: list[str]):
        """
        Fit on combined corpus (resumes + jobs).
        """
        self.vectorizer.fit(texts)

    def transform(self, texts: list[str]):
        """
        Transform new texts using fitted vectorizer.
        """
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts: list[str]):
        """
        Convenience method.
        """
        return self.vectorizer.fit_transform(texts)

# Singleton pattern to ensure only one instance of the embedding service is created
_instance: TFIDFEmbeddingService | None = None

def load_vectorizer(path="tfidf_vectorizer.pkl") -> TFIDFEmbeddingService:
    """Load the fitted vectorizer from disk, caching it for the app lifetime."""
    global _instance
    if _instance is None:
        with open(path, "rb") as f:
            vectorizer = pickle.load(f)
        service = TFIDFEmbeddingService()
        service.vectorizer = vectorizer
        _instance = service
    return _instance