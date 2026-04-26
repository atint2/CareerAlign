"""
One-time script to fit the TF-IDF vectorizer on the combined corpus
and save it to disk. Run this during setup, not at runtime.

Usage:
    python -m backend.services.fit_tf_idf_vectorizer
"""

import pickle
from backend.app import database
from backend.app.services.tf_idf_embedder import TFIDFEmbeddingService
from backend.app import models

def fit_and_save_vectorizer():
    # Load env
    from dotenv import load_dotenv
    load_dotenv()

    # Initialize database
    database.init_db()

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

if __name__ == "__main__":
    fit_and_save_vectorizer()