import pickle
from google.genai import Client
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import os

# Load API key
from dotenv import load_dotenv
import os
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
client = Client(api_key=API_KEY)

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
        [
            f"[TFIDF] {job['title']} | similarity={job['similarity']:.4f}\n{job['snippet']}"
            for job in top_jobs_tfidf
        ] +
        [
            f"[SBERT] {job['title']} | similarity={job['similarity']:.4f}\n{job['snippet']}"
            for job in top_jobs_sbert
        ]
    )

    prompt = f"""
        You are an expert AI career advisor.

        Your task is to analyze the resume and the retrieved job descriptions.

        Instructions:
        - Choose ONLY one job from the provided list.
        - Do NOT invent new job titles.
        - Base your reasoning on skills, responsibilities, and experience alignment.
        - If confidence is below 70, suggest a better role based strictly on the resume.

        Return ONLY valid JSON.
        No explanations outside JSON.
        No markdown formatting.
        No backticks.

        Required JSON structure:

        {{
        "recommended_job_title": "string",
        "confidence_score": number (0-100),
        "match_summary": "2-4 sentence explanation referencing specific skills",
        "improvement_suggestions": "Specific skill gaps or improvements",
        "alternative_role": "string or null"
        }}

        RESUME: {resume_text} 

        TOP MATCHING JOB DESCRIPTIONS: {job_snippets} 
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

        text = response.text.strip()

        if text.startswith("```"):
            text = text.split("```")[1]  # get content inside fences
            text = text.replace("json", "", 1).strip()

        return text
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

    # Find matches using SBERT
    top_jobs_sbert = find_top_job_matches_sbert(resume_text_sbert, sbert_service, db_session, models, top_n=3)

    # Create LLM prompt
    prompt = create_llm_prompt(resume_text, top_jobs_tfidf, top_jobs_sbert)
    # Generate insights using LLM
    insights_text = generate_resume_insights(prompt)
    print(insights_text)
    insights = json.loads(insights_text)

    return {
        "tfidf_matches": top_jobs_tfidf,
        "sbert_matches": top_jobs_sbert,
        "insights": insights
    }