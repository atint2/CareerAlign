from typing import Optional, List, Dict, Any
import json
from backend.app.matcher.match_resume import find_top_job_matches_tfidf, find_top_job_matches_sbert, create_llm_prompt, generate_resume_insights, normalize_array, rank_jobs_within_clusters
from backend.app.services.tf_idf_embedder import load_vectorizer
from backend.app.services.sbert_embedder import get_sbert_service
from backend.app import models
from data.scripts.preprocessor_tfidf import TFIDFPreprocessor
from data.scripts.preprocessor_sbert import SBERTPreprocessor
# import matplotlib.pyplot as plt

def hybrid_rank_jobs(tfidf_matches, sbert_matches, alpha=0.75):

    tfidf_dict = {job["cluster_id"]: job for job in tfidf_matches}
    sbert_dict = {job["cluster_id"]: job for job in sbert_matches}

    # Blend jobs that appear in result sets for a hybrid score
    all_cluster_ids = sorted(set(tfidf_dict.keys()).union(sbert_dict.keys()))

    # Build aligned scores list
    tfidf_scores = []
    sbert_scores = []
    for cid in all_cluster_ids:
        tfidf_scores.append(tfidf_dict.get(cid, {}).get("similarity", 0))
        sbert_scores.append(sbert_dict.get(cid, {}).get("similarity", 0))

    # Normalize scores
    tfidf_norm = normalize_array(tfidf_scores)
    sbert_norm = normalize_array(sbert_scores) 


    # Compute hybrid score per cluster
    hybrid_results = []
    for i, cid in enumerate(all_cluster_ids):
        tfidf_score_raw = tfidf_scores[i]
        sbert_score_raw = sbert_scores[i]

        tfidf_score_norm = tfidf_norm[i]
        sbert_score_norm = sbert_norm[i]

        hybrid_score = alpha * sbert_score_norm + (1 - alpha) * tfidf_score_norm

        # Safely get base job info
        base_job = sbert_dict.get(cid) or tfidf_dict.get(cid)

        hybrid_results.append({
            **base_job,
            "tfidf_similarity": tfidf_score_raw,
            "sbert_similarity": sbert_score_raw,
            "similarity": hybrid_score,
            "hybrid_score": hybrid_score,
            "hybrid_percent": round(hybrid_score * 100, 1)
        })

    # Sort descending
    hybrid_results.sort(key=lambda x: x["hybrid_score"], reverse=True)

    return hybrid_results

def hybrid_match(resume_text: str, job_desc: Optional[str], db_session):
    """Match resumes to LLM-generated job descriptions using hybrid approach -- combining pre-trained SBERT model and trained TF-IDF model."""

    # Load embedding services
    try:
        tfidf_service = load_vectorizer()
        sbert_service = get_sbert_service()
    except Exception as e:
        raise RuntimeError(f"Failed to load embedding services: {e}") from e

    # Initialize preprocessors
    try:
        tfidf_prep = TFIDFPreprocessor()
        sbert_prep = SBERTPreprocessor()
    except Exception as e:
        raise RuntimeError(f"Failed to load preprocessors: {e}") from e

    # Preprocess resume text
    resume_text_tfidf = tfidf_prep.clean_text_tfidf(resume_text)
    resume_text_sbert = sbert_prep.clean_text_sbert(resume_text)

    job_desc_tfidf = None
    job_desc_sbert = None
    # Preprocess custom job description (if provided)
    if job_desc:
        job_desc_tfidf = tfidf_prep.clean_text_tfidf(job_desc)
        job_desc_sbert = sbert_prep.clean_text_sbert(job_desc)

    # Find matches using TF-IDF
    top_jobs_tfidf = find_top_job_matches_tfidf(
        resume_text_tfidf,
        tfidf_service,
        db_session,
        models,
        top_n=20,
        job_desc_text=job_desc_tfidf
    )

    # Find matches using SBERT
    top_jobs_sbert = find_top_job_matches_sbert(
        resume_text_sbert,
        sbert_service,
        db_session,
        models,
        top_n=20,
        job_desc_text=job_desc_sbert
    )

    hybrid_matches = hybrid_rank_jobs(
        top_jobs_tfidf,
        top_jobs_sbert
    )
    # Create LLM prompt
    prompt = create_llm_prompt(resume_text, top_jobs_hybrid=hybrid_matches[:10]
)
    # Generate insights using LLM
    try:
        insights_text = generate_resume_insights(prompt)
        insights = json.loads(insights_text)
    except RuntimeError as e:
        print(f"Insights unavailable: {e}")
        insights = None
    except json.JSONDecodeError as e:
        print(f"Failed to parse insights JSON: {e}")
        insights = None

    # # For score distribution histogram
    # # Extract scores
    # tfidf_scores_raw = [job["similarity"] for job in top_jobs_tfidf]
    # sbert_scores_raw = [job["similarity"] for job in top_jobs_sbert]
    # tfidf_scores = normalize_array(tfidf_scores_raw)
    # sbert_scores = normalize_array(sbert_scores_raw)
    # hybrid_scores = [job["hybrid_score"] for job in hybrid_matches]
    # plt.figure()

    # plt.hist(tfidf_scores, bins=20, alpha=0.5, label="TF-IDF")
    # plt.hist(sbert_scores, bins=20, alpha=0.5, label="SBERT")
    # plt.hist(hybrid_scores, bins=20, alpha=0.5, label="Hybrid")

    # plt.xlabel("Normalized Similarity Score")
    # plt.ylabel("Frequency")
    # plt.title("Comparison of Matching Score Distributions")

    # plt.legend()

    # plt.savefig("model_comparison_hist.png", dpi=300, bbox_inches='tight')
    # plt.close()

    return {
        "tfidf_matches": top_jobs_tfidf,
        "sbert_matches": top_jobs_sbert,
        "hybrid_matches": hybrid_matches[:10],
        "insights": insights
    }

def downstream_match(resume_text: str, hybrid_matches: List[Dict[str, Any]], db_session):
    """Optional matching of resumes to job postings in database given matched cluster ids."""

    # Load embedding services
    try:
        tfidf_service = load_vectorizer()
        sbert_service = get_sbert_service()
    except Exception as e:
        raise RuntimeError(f"Failed to load embedding services: {e}") from e

    # Initialize preprocessors
    try:
        tfidf_prep = TFIDFPreprocessor()
        sbert_prep = SBERTPreprocessor()
    except Exception as e:
        raise RuntimeError(f"Failed to load preprocessors: {e}") from e

    # Preprocess resume text
    resume_text_tfidf = tfidf_prep.clean_text_tfidf(resume_text)
    resume_text_sbert = sbert_prep.clean_text_sbert(resume_text)

    # Rank individual postings within matched clusters
    posting_matches = rank_jobs_within_clusters(
            resume_text=resume_text,
            resume_text_tfidf=resume_text_tfidf,
            resume_text_sbert=resume_text_sbert,
            matched_clusters=hybrid_matches,
            tfidf_service=tfidf_service,
            sbert_service=sbert_service,
            db_session=db_session,
            models=models
        )
     # Create LLM prompt
    prompt = create_llm_prompt(resume_text, top_jobs_hybrid=posting_matches)
    # Generate insights using LLM
    try:
        insights_text = generate_resume_insights(prompt)
        insights = json.loads(insights_text)
    except RuntimeError as e:
        print(f"Insights unavailable: {e}")
        insights = None
    except json.JSONDecodeError as e:
        print(f"Failed to parse insights JSON: {e}")
        insights = None

    return {
        "posting_matches": posting_matches,
        "insights": insights
    }