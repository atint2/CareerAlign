import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import sys
import os

PROCESSED_DIR = Path(__file__).parents[1] / "services" / "processed_resumes"

def setup_backend_imports(path="backend"):
	# Ensure backend/ is on sys.path so its modules import as top-level modules
	root = Path(__file__).resolve().parents[2]
	backend_dir = root / path
	sys.path.insert(0, str(backend_dir))

def load_vectorizer(path="tfidf_vectorizer.pkl"):
    
    with open(path, "rb") as f:
        vectorizer = pickle.load(f)
    # Wrap into your embedding service
    setup_backend_imports("backend/services")
    from tf_idf_embedder import TFIDFEmbeddingService
    embedding_service = TFIDFEmbeddingService()
    embedding_service.vectorizer = vectorizer
    return embedding_service

def find_top_job_matches_tfidf(resume_text, embedding_service, db_session, models, top_n=3):
    # Transform resume
    resume_vector = embedding_service.transform([resume_text]).toarray()  # (1, vocab_size)

    # Load all job embeddings from cluster_embeddings table
    job_embeddings = db_session.query(models.ClusterEmbeddingTFIDF).all()
    if not job_embeddings:
        return []

    X_jobs = np.array([je.embedding for je in job_embeddings])
    cluster_ids = [je.cluster_id for je in job_embeddings]

    # Compute cosine similarity
    similarities = cosine_similarity(resume_vector, X_jobs).flatten()

    # Get top N matches
    top_indices = similarities.argsort()[::-1][:top_n]

    top_matches = []
    for idx in top_indices:
        cluster = db_session.query(models.Cluster).filter(models.Cluster.id == cluster_ids[idx]).first()
        top_matches.append({
            "cluster_id": cluster.id,
            "title": cluster.title,
            "similarity": float(similarities[idx]),
            "snippet": cluster.general_job_desc_raw[:200] + "..."
        })
    return top_matches

def find_top_job_matches_sbert(resume_text, sbert_service, db_session, models, top_n=3):
    # Embed resume using SBERT
    resume_embedding = sbert_service.embed([resume_text])

    # Load all cluster embeddings from database
    cluster_embeddings = db_session.query(models.ClusterEmbeddingSBERT).all()
    if not cluster_embeddings:
        return []
    
    X_jobs = np.array([ce.embedding for ce in cluster_embeddings])
    cluster_ids = [ce.cluster_id for ce in cluster_embeddings]
    
    # Compute cosine similarity
    similarities = cosine_similarity(resume_embedding, X_jobs).flatten()

    # Get top N matches
    top_indices = similarities.argsort()[::-1][:top_n]

    top_matches = []
    for idx in top_indices:
        cluster = db_session.query(models.Cluster).filter(models.Cluster.id == cluster_ids[idx]).first()
        top_matches.append({
            "cluster_id": cluster.id,
            "title": cluster.title,
            "similarity": float(similarities[idx]),
            "snippet": cluster.general_job_desc_raw[:200] + "..."
        })
    return top_matches

def main():
    setup_backend_imports()

    try:
        import database
        import models
    except Exception as e:
        print("Exception importing backend modules:", e)

    # Initialize database session
    SessionLocal = database.SessionLocal
    db_session = SessionLocal()
    
    embedding_service = load_vectorizer("tfidf_vectorizer.pkl")

    # Example resume text
    for filename in os.listdir(PROCESSED_DIR):
            # Read file
            file_path = PROCESSED_DIR / filename
            with open(file_path, "r", encoding="utf-8") as f:
                resume_text = f.read()
            print(f"\nFinding matches for resume: {filename}")

            setup_backend_imports("data/scripts")
            from preprocessor_tfidf import TFIDFPreprocessor
            tfidf_prep = TFIDFPreprocessor()
            from preprocessor_sbert import SBERTPreprocessor
            sbert_prep = SBERTPreprocessor()

            resume_text_tfidf = tfidf_prep.clean_text_tfidf(resume_text)
            resume_text_sbert = sbert_prep.clean_text_sbert(resume_text)

            setup_backend_imports("backend/services")
            from sbert_embedder import SBERTEmbeddingService
            sbert_service = SBERTEmbeddingService()

            # First use TF-IDF to find top job matches
            print(f"\nFinding matches for resume: {filename} using TF-IDF...")
            top_jobs = find_top_job_matches_tfidf(resume_text_tfidf, embedding_service, db_session, models, top_n=3)
            for job in top_jobs:
                print(f"{job['title']} - similarity: {job['similarity']:.4f}")
                print(job["snippet"])
                print("-" * 60)

            # Then use SBERT to find top cluster matches
            print(f"\nFinding matches for resume: {filename} using SBERT...")
            top_jobs = find_top_job_matches_sbert(resume_text_sbert, sbert_service, db_session, models, top_n=3)
            for job in top_jobs:
                print(f"{job['title']} - similarity: {job['similarity']:.4f}")
                print(job["snippet"])
                print("-" * 60)

    db_session.close()

if __name__ == "__main__":
    main()