from typing import Optional

from backend.matcher.match_resume import find_top_job_matches_tfidf, find_top_job_matches_sbert
from backend.services.fit_tf_idf_vectorizer import load_vectorizer, find_top_keywords

def hybrid_rank_jobs(tfidf_matches, sbert_matches, alpha=0.65):

    tfidf_dict = {job["cluster_id"]: job for job in tfidf_matches}
    sbert_dict = {job["cluster_id"]: job for job in sbert_matches}

    # Normalize TF-IDF scores to [0, 1] so they're on the same scale as SBERT
    tfidf_scores = [job["similarity"] for job in tfidf_matches]
    tfidf_max = max(tfidf_scores) if tfidf_scores else 1.0
    tfidf_min = min(tfidf_scores) if tfidf_scores else 0.0
    tfidf_range = tfidf_max - tfidf_min or 1.0  # avoid division by zero

    def normalize_tfidf(score):
        return (score - tfidf_min) / tfidf_range

    # Only blend jobs that appear in BOTH result sets for a genuine hybrid score
    shared_cluster_ids = set(tfidf_dict.keys()).intersection(sbert_dict.keys())

    hybrid_results = []

    for cid in shared_cluster_ids:
        tfidf_score_raw = tfidf_dict[cid]["similarity"]
        sbert_score = sbert_dict[cid]["similarity"]
        tfidf_score_norm = normalize_tfidf(tfidf_score_raw)

        hybrid_score = alpha * sbert_score + (1 - alpha) * tfidf_score_norm

        hybrid_results.append({
            **sbert_dict[cid],
            "tfidf_similarity": tfidf_score_raw,
            "sbert_similarity": sbert_score,
            "similarity": hybrid_score,   # overwrite so render_match_section displays hybrid %
            "hybrid_score": hybrid_score,
            "hybrid_percent": round(hybrid_score * 100, 1)
        })

    hybrid_results.sort(key=lambda x: x["hybrid_score"], reverse=True)

    return hybrid_results

def hybrid_match(resume_text: str, job_desc: Optional[str], db_session):
    from backend import models

    try:
        tfidf_service = load_vectorizer("tfidf_vectorizer.pkl")
        from backend.services.sbert_embedder import SBERTEmbeddingService
        sbert_service = SBERTEmbeddingService()
    except Exception as e:
        raise RuntimeError(f"Failed to load embedding services: {e}") from e

    try:
        from data.scripts.preprocessor_tfidf import TFIDFPreprocessor
        tfidf_prep = TFIDFPreprocessor()
        from data.scripts.preprocessor_sbert import SBERTPreprocessor
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
        top_n=10,
        job_desc_text=job_desc_tfidf
    )

    # Find matches using SBERT
    top_jobs_sbert = find_top_job_matches_sbert(
        resume_text_sbert,
        sbert_service,
        db_session,
        models,
        top_n=10,
        job_desc_text=job_desc_sbert
    )

    hybrid_matches = hybrid_rank_jobs(
        top_jobs_tfidf,
        top_jobs_sbert
    )[:5]

    return {
        "tfidf_matches": top_jobs_tfidf[:5],
        "sbert_matches": top_jobs_sbert[:5],
        "hybrid_matches": hybrid_matches
    }