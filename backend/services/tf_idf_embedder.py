"""
Service for embedding documents using TF-IDF
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text

class TFIDFEmbeddingService:
    """
    TF-IDF embedding service for resumes and job descriptions.
    """

    def __init__(self):
        custom_stop_words = list(
            text.ENGLISH_STOP_WORDS.union({"occasionally", "usually"})
        )

        self.vectorizer = TfidfVectorizer(
            stop_words=custom_stop_words,
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