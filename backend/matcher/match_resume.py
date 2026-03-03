import pickle
from google.genai import Client
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import sys
import os

PROCESSED_DIR = Path(__file__).parents[1] / "services" / "processed_resumes"

# Load API key
from dotenv import load_dotenv
import os
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
client = Client(api_key=API_KEY)

def setup_backend_imports(path="backend"):
	# Ensure backend/ is on sys.path so its modules import as top-level modules
	root = Path(__file__).resolve().parents[2]
	backend_dir = root / path
	sys.path.insert(0, str(backend_dir))

def load_vectorizer(path="tfidf_vectorizer.pkl"):
    
    with open(path, "rb") as f:
        vectorizer = pickle.load(f)
    # Wrap into your embedding service
    from backend.services.tf_idf_embedder import TFIDFEmbeddingService
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

def create_llm_prompt(resume_text, top_jobs_tfidf, top_jobs_sbert):
    job_snippets = "\n\n".join(
        [f"{job['title']} (TF-IDF similarity: {job['similarity']:.4f}): {job['snippet']}" for job in top_jobs_tfidf] +
        [f"{job['title']} (SBERT similarity: {job['similarity']:.4f}): {job['snippet']}" for job in top_jobs_sbert]
    )

    prompt = f"""
    Given the following resume and top matching job descriptions, identify the most likely job role the candidate is targeting, and provide a concise summary of their qualifications that match with a confidence score. If the confidence is low, provide insights on what role would be a better fit based on the resume content.

    RESUME:
    {resume_text}

    TOP MATCHING JOB DESCRIPTIONS:
        {job_snippets}
        """
    
    return prompt.strip()
    
def generate_resume_insights(prompt):
    """
    Call LLM to generate a generalized job description.
    """
    if not prompt:
        return "No prompt provided for LLM generation."

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
        )
        return response.text.strip()
    except Exception as e:
        print("LLM generation failed:", e)
        return "Feedback unavailable."

def match_resume(resume_text: str, db_session):
    from backend import models

    # Load embedding services
    try:
        tfidf_service = load_vectorizer("tfidf_vectorizer.pkl")
        from backend.services.sbert_embedder import SBERTEmbeddingService
        sbert_service = SBERTEmbeddingService()
    except Exception as e:
        print("Exception loading embedding services:", e)
    
    # Load preprocessors
    try:
        from data.scripts.preprocessor_tfidf import TFIDFPreprocessor
        tfidf_prep = TFIDFPreprocessor()
        from data.scripts.preprocessor_sbert import SBERTPreprocessor
        sbert_prep = SBERTPreprocessor()
    except Exception as e:
        print("Exception loading preprocessors:", e)
    
    # Preprocess resume text
    resume_text_tfidf = tfidf_prep.clean_text_tfidf(resume_text)
    resume_text_sbert = sbert_prep.clean_text_sbert(resume_text)
    
    # Find matches using TF-IDF
    top_jobs_tfidf = find_top_job_matches_tfidf(resume_text_tfidf, tfidf_service, db_session, models, top_n=3)
    for job in top_jobs_tfidf:
        print(f"{job['title']} - similarity: {job['similarity']:.4f}")
        print(job["snippet"])
        print("-" * 60)

    # Find matches using SBERT
    top_jobs_sbert = find_top_job_matches_sbert(resume_text_sbert, sbert_service, db_session, models, top_n=3)
    for job in top_jobs_sbert:
        print(f"{job['title']} - similarity: {job['similarity']:.4f}")
        print(job["snippet"])
        print("-" * 60)

    return {
        "tfidf_matches": top_jobs_tfidf,
        "sbert_matches": top_jobs_sbert
    }

# def main():
#     setup_backend_imports()

#     try:
#         import database
#         import models
#     except Exception as e:
#         print("Exception importing backend modules:", e)

#     # Initialize database session
#     SessionLocal = database.SessionLocal
#     db_session = SessionLocal()
    
#     embedding_service = load_vectorizer("tfidf_vectorizer.pkl")

#     # Example resume text
#     for filename in os.listdir(PROCESSED_DIR):
#             # Read file
#             file_path = PROCESSED_DIR / filename
#             with open(file_path, "r", encoding="utf-8") as f:
#                 resume_text = f.read()
#             print(f"\nFinding matches for resume: {filename}")

#             setup_backend_imports("data/scripts")
#             from preprocessor_tfidf import TFIDFPreprocessor
#             tfidf_prep = TFIDFPreprocessor()
#             from preprocessor_sbert import SBERTPreprocessor
#             sbert_prep = SBERTPreprocessor()

#             resume_text_tfidf = tfidf_prep.clean_text_tfidf(resume_text)
#             resume_text_sbert = sbert_prep.clean_text_sbert(resume_text)

#             setup_backend_imports("backend/services")
#             from sbert_embedder import SBERTEmbeddingService
#             sbert_service = SBERTEmbeddingService()

#             # First use TF-IDF to find top job matches
#             print(f"\nFinding matches for resume: {filename} using TF-IDF...")
#             top_jobs_tfidf = find_top_job_matches_tfidf(resume_text_tfidf, embedding_service, db_session, models, top_n=3)
#             for job in top_jobs_tfidf:
#                 print(f"{job['title']} - similarity: {job['similarity']:.4f}")
#                 print(job["snippet"])
#                 print("-" * 60)

#             # Then use SBERT to find top cluster matches
#             print(f"\nFinding matches for resume: {filename} using SBERT...")
#             top_jobs_sbert = find_top_job_matches_sbert(resume_text_sbert, sbert_service, db_session, models, top_n=3)
#             for job in top_jobs_sbert:
#                 print(f"{job['title']} - similarity: {job['similarity']:.4f}")
#                 print(job["snippet"])
#                 print("-" * 60)

#             # Create LLM prompt and generate insights
#             if (filename == "Masters Resume June 2024.txt"):
#                 prompt = create_llm_prompt(resume_text, top_jobs_tfidf, top_jobs_sbert)
#                 insights = generate_resume_insights(prompt)
#                 print(f"\nLLM-Generated Insights for {filename}:\n{insights}")


#     db_session.close()

# if __name__ == "__main__":
#     main()