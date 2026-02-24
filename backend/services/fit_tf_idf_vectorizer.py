import pickle
from tf_idf_embedder import TFIDFEmbeddingService
from pathlib import Path
import sys

def setup_backend_imports():
	# Ensure backend/ is on sys.path so its modules import as top-level modules
	root = Path(__file__).resolve().parents[2]
	backend_dir = root / "backend"
	sys.path.insert(0, str(backend_dir))

def fit_and_save_vectorizer():
    setup_backend_imports()
    try:
        import database
        import models
    except Exception as e:
        print("Exception importing backend modules:", e)

    # Initialize database session
    SessionLocal = database.SessionLocal
    db_session = SessionLocal()
    try:
        # Fetch all job descriptions and resumes
        job_descs = db_session.query(models.Cluster).filter(models.Cluster.general_job_desc_raw != None).all()
        resume_texts = db_session.query(models.Resume).filter(models.Resume.content_raw != None).all()

        # Combine texts
        combined_texts = [jd.general_job_desc_raw for jd in job_descs] + [r.content_raw for r in resume_texts]

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