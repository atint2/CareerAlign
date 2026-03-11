from google.genai import Client
from backend.services.fit_tf_idf_vectorizer import load_vectorizer, find_top_keywords
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load API key
from dotenv import load_dotenv
import os
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
client = Client(api_key=API_KEY)

def find_top_job_matches_tfidf(resume_text, embedding_service, db_session, models, top_n=3, job_desc_text=None):
    # Transform resume
    resume_vector = embedding_service.transform([resume_text]).toarray()

    # If job description text is provided, embed it and compute similarity
    if job_desc_text:
        job_desc_vector = embedding_service.transform([job_desc_text]).toarray()
        similarity = cosine_similarity(resume_vector, job_desc_vector).flatten()[0]
        top_keywords = find_top_keywords(job_desc_text, resume_text)
        return similarity, top_keywords
        
    # Load all job embeddings from cluster_embeddings table
    job_embeddings = db_session.query(models.ClusterEmbeddingTFIDF).all()
    if not job_embeddings:
        return []

    X_jobs = np.vstack([np.array(je.embedding, dtype=float) for je in job_embeddings])
    cluster_ids = [je.cluster_id for je in job_embeddings]

    # Compute cosine similarity
    similarities = cosine_similarity(resume_vector, X_jobs).flatten()

    # Get top N matches
    top_indices = similarities.argsort()[::-1][:top_n]

    top_matches = []
    for idx in top_indices:
        cluster = db_session.query(models.Cluster).filter(models.Cluster.id == cluster_ids[idx]).first()
        top_keywords = find_top_keywords(cluster.general_job_desc_tfidf, resume_text)
        top_matches.append({
            "cluster_id": cluster.id,
            "title": cluster.title,
            "description": cluster.general_job_desc_raw,
            "similarity": float(similarities[idx]),
            "similarity_percent": round(similarities[idx] * 100, 1),
            "snippet": cluster.general_job_desc_raw[:200] + "...",
            "top_keywords": top_keywords
        })
    return top_matches

def find_top_job_matches_sbert(resume_text, sbert_service, db_session, models, top_n=3, job_desc_text=None):
    # Embed resume using SBERT
    resume_embedding = sbert_service.embed([resume_text])
    # If job description text is provided, embed it and compute similarity
    if job_desc_text:
        job_desc_embedding = sbert_service.embed([job_desc_text])
        similarity = cosine_similarity(resume_embedding, job_desc_embedding).flatten()[0]
        top_keywords = find_top_keywords(job_desc_text, resume_text)
        return similarity, top_keywords

    # Load all cluster embeddings from database
    cluster_embeddings = db_session.query(models.ClusterEmbeddingSBERT).all()
    if not cluster_embeddings:
        return []
    
    X_jobs = np.vstack([np.array(ce.embedding, dtype=float) for ce in cluster_embeddings])
    cluster_ids = [ce.cluster_id for ce in cluster_embeddings]
    
    # Compute cosine similarity
    similarities = cosine_similarity(resume_embedding, X_jobs).flatten()

    # Get top N matches
    top_indices = similarities.argsort()[::-1][:top_n]

    top_matches = []
    for idx in top_indices:
        cluster = db_session.query(models.Cluster).filter(models.Cluster.id == cluster_ids[idx]).first()
        top_keywords = find_top_keywords(cluster.general_job_desc_tfidf, resume_text)
        top_matches.append({
            "cluster_id": cluster.id,
            "title": cluster.title,
            "description": cluster.general_job_desc_raw,
            "similarity": float(similarities[idx]),
            "similarity_percent": round(similarities[idx] * 100, 1),
            "snippet": cluster.general_job_desc_raw[:200] + "...",
            "top_keywords": top_keywords 
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
        - Base your reasoning on skills, responsibilities, and experience alignment.
        - If there is a clear match, provide a confidence score between 0 and 100. Follow this scoring logic:
            90-100: Nearly identical titles, 90%+ skill overlap, and sufficient years of experience.
            70-89: Strong match, but perhaps missing one "nice-to-have" skill or coming from a slightly different industry.
            50-69: Partial match; the candidate has the foundation but would require significant upskilling.
            Below 50: Major gaps in core requirements.
        - If confidence_score is below 70, you MUST suggest an alternative role under "alternative_role" that better fits the candidate's profile based on their resume. This is REQUIRED, not optional.
        - If confidence_score is 70 or above, set "alternative_role" to null.
        - Provide a 2-4 sentence explanation referencing specific skills or experiences that led to your recommendation.
        - If you assign an alternative role, you MUST give an additional 2-4 sentence explanation for the alternative match under "alternative_role_suggestions". If you do not provide an alternative role, set "alternative_role_suggestions" to null.
        
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
        "alternative_role": "string or null",
        "alternative_role_suggestions": "string or null"
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

def match_resume(resume_text: str, job_desc: str | None, db_session):
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

    if job_desc:

        job_desc_tfidf = tfidf_prep.clean_text_tfidf(job_desc)
        job_desc_sbert = sbert_prep.clean_text_sbert(job_desc)

        tfidf_similarity, tfidf_keywords = find_top_job_matches_tfidf(
            resume_text_tfidf,
            tfidf_service,
            db_session,
            models,
            job_desc_text=job_desc_tfidf
        )

        sbert_similarity, sbert_keywords = find_top_job_matches_sbert(
            resume_text_sbert,
            sbert_service,
            db_session,
            models,
            job_desc_text=job_desc_sbert
        )

        return {
            "tfidf_matches": [{
                "similarity": float(tfidf_similarity),
                "top_keywords": tfidf_keywords
            }],
            "sbert_matches": [{
                "similarity": float(sbert_similarity),
                "top_keywords": sbert_keywords
            }]
        }
    
    # If no custom job description provided, proceed with normal matching against clusters
    
    # Find matches using TF-IDF
    top_jobs_tfidf = find_top_job_matches_tfidf(resume_text_tfidf, tfidf_service, db_session, models, top_n=5)

    # Find matches using SBERT
    top_jobs_sbert = find_top_job_matches_sbert(resume_text_sbert, sbert_service, db_session, models, top_n=5)

    # # Create LLM prompt
    # prompt = create_llm_prompt(resume_text, top_jobs_tfidf, top_jobs_sbert)
    # # Generate insights using LLM
    # insights_text = generate_resume_insights(prompt)
    # print(insights_text)
    # insights = json.loads(insights_text)

    return {
        "tfidf_matches": top_jobs_tfidf,
        "sbert_matches": top_jobs_sbert,
        # "insights": insights,
    }