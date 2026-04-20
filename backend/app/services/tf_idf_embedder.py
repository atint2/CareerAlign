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

# Keyword utility functions for resume optimization
def find_top_keywords(job_desc, resume_text):
    """Find top keywords in job description that are most relevant to the resume."""
    
    embedding_service = load_vectorizer()
    vectorizer = embedding_service.vectorizer

    job_desc_vec = vectorizer.transform([job_desc])
    resume_vec = vectorizer.transform([resume_text])

    # Element-wise multiply to get shared importance
    shared_scores = job_desc_vec.multiply(resume_vec)

    # Convert to array
    scores = shared_scores.toarray().flatten()

    feature_names = vectorizer.get_feature_names_out()

    # Get top scoring features
    top_indices = scores.argsort()[::-1]

    top_keywords = [
        feature_names[idx] for idx in top_indices
        if scores[idx] > 0 and feature_names[idx] not in CUSTOM_STOPWORDS
    ]

    return top_keywords[:10]

def find_missing_keywords(job_desc, resume_text):
    """Find top keywords in job description that are absent from the resume."""
    
    embedding_service = load_vectorizer()
    vectorizer = embedding_service.vectorizer

    job_desc_vec = vectorizer.transform([job_desc]).toarray().flatten()
    resume_vec = vectorizer.transform([resume_text]).toarray().flatten()

    feature_names = vectorizer.get_feature_names_out()

    # Keep only terms that appear in the job desc but not in the resume
    missing_scores = np.where(resume_vec == 0, job_desc_vec, 0)

    top_indices = missing_scores.argsort()[::-1]

    missing_keywords = [
        feature_names[idx] for idx in top_indices
        if missing_scores[idx] > 0 and feature_names[idx] not in CUSTOM_STOPWORDS
    ]

    return missing_keywords[:10]

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