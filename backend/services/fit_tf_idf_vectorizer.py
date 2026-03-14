import pickle
from backend.services.tf_idf_embedder import TFIDFEmbeddingService
from backend import database, models
from backend.config import CUSTOM_STOPWORDS
import numpy as np

def fit_and_save_vectorizer():
    # Initialize database session
    SessionLocal = database.SessionLocal
    db_session = SessionLocal()
    try:
        # Fetch all job descriptions and resumes
        job_descs = db_session.query(models.Cluster).filter(models.Cluster.general_job_desc_tfidf != None).all()
        resume_texts = db_session.query(models.Resume).filter(models.Resume.content_raw != None).all()

        # Combine texts
        combined_texts = [jd.general_job_desc_tfidf for jd in job_descs] + [r.content_raw for r in resume_texts]

        # Fit TF-IDF vectorizer
        embedding_service = TFIDFEmbeddingService()
        embedding_service.fit_transform(combined_texts)

        # Save vectorizer to disk
        with open("tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(embedding_service.vectorizer, f)

        print("TF-IDF vectorizer fitted and saved to tfidf_vectorizer.pkl")

    finally:
        db_session.close()

def load_vectorizer(path="tfidf_vectorizer.pkl"):
    with open(path, "rb") as f:
        vectorizer = pickle.load(f)
    # Wrap into your embedding service
    from backend.services.tf_idf_embedder import TFIDFEmbeddingService
    embedding_service = TFIDFEmbeddingService()
    embedding_service.vectorizer = vectorizer
    return embedding_service

def find_top_keywords(job_desc, resume_text, top_k=20):
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
    top_indices = scores.argsort()[-top_k:][::-1]

    top_keywords = [
        feature_names[idx] for idx in top_indices
        if scores[idx] > 0 and feature_names[idx] not in CUSTOM_STOPWORDS
    ]

    return top_keywords

def find_missing_keywords(job_desc, resume_text, top_k=10):
    """Find top keywords in job description that are absent from the resume."""
    
    embedding_service = load_vectorizer()
    vectorizer = embedding_service.vectorizer

    job_desc_vec = vectorizer.transform([job_desc]).toarray().flatten()
    resume_vec = vectorizer.transform([resume_text]).toarray().flatten()

    feature_names = vectorizer.get_feature_names_out()

    # Keep only terms that appear in the job desc but not in the resume
    missing_scores = np.where(resume_vec == 0, job_desc_vec, 0)

    top_indices = missing_scores.argsort()[-top_k:][::-1]

    missing_keywords = [
        feature_names[idx] for idx in top_indices
        if missing_scores[idx] > 0 and feature_names[idx] not in CUSTOM_STOPWORDS
    ]

    return missing_keywords

if __name__ == "__main__":
    fit_and_save_vectorizer()