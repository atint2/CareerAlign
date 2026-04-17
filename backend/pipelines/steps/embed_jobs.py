# This script defines the pipeline step for embedding job descriptions using both SBERT and TF-IDF methods
# Dimensionality reduction must be ran after this step since it reduces the dimensionality of the generated SBERT embeddings which are used for clustering

import numpy as np
import backend.app.models as models
from backend.app.config import EMBEDDING_MODEL
from backend.app.services.sbert_embedder import SBERTEmbeddingService
from backend.app.services.fit_tf_idf_vectorizer import load_vectorizer
from typing import Optional

def save_embeddings(job_ids: list[int], embeddings: np.ndarray, model: str, db_session, model_version: Optional[str] = None):
    if len(job_ids) != len(embeddings):
        raise ValueError("job_ids and embeddings must have the same length")

    if model == "SBERT":
        try:
            embedding_objs = [models.JobEmbeddingSBERT(
                    embedding=vector.tolist(),
                    model_version=model_version,
                    job_posting_id=job_id
            )
            for job_id, vector in zip(job_ids, embeddings)
            ]

            db_session.add_all(embedding_objs)  # bulk insert
            db_session.commit()

            print(f"Saved {len(job_ids)} SBERT embeddings to the database.")

        except Exception as e:
            db_session.rollback()
            print("Exception during SBERT embedding DB insertion:", e)

    elif model == "TF-IDF":
        try:
            embedding_objs = [
                models.JobEmbeddingTFIDF(
                    embedding=vector.tolist(),
                    job_posting_id=job_id
                )
                for job_id, vector in zip(job_ids, embeddings)
            ]

            db_session.add_all(embedding_objs)  # bulk insert
            db_session.commit()

            print(f"Saved {len(job_ids)} TF-IDF embeddings to the database.")
            
        except Exception as e:
            db_session.rollback()
            print("Exception during TF-IDF embedding DB insertion:", e)

def run(db_session):
    try:
        # Fetch all job postings with SBERT descriptions that haven't been embedded yet
        job_postings = (
            db_session.query(models.JobPosting)
            .outerjoin(
                models.JobEmbeddingSBERT,
                models.JobPosting.id == models.JobEmbeddingSBERT.job_posting_id
            )
            .filter(
                models.JobPosting.desc_sbert.isnot(None),
                models.JobEmbeddingSBERT.job_posting_id.is_(None)
            )
            .all()
        )

        if not job_postings:
            print("No job postings found with SBERT descriptions that need embedding.")
        
        else:
        
            job_ids = [jp.id for jp in job_postings]
            job_descriptions = [jp.desc_sbert for jp in job_postings]

            print(f"Embedding {len(job_descriptions)} job descriptions using SBERT...")
            embedding_service = SBERTEmbeddingService()
            embeddings = embedding_service.embed(job_descriptions)

            print("Saving SBERT embeddings to database...")
            save_embeddings(job_ids, embeddings, EMBEDDING_MODEL, db_session)

    except Exception as e:
        print("Exception during SBERT embedding:", e)
    
    try:
        # Fetch all job postings with TF-IDF descriptions that haven't been embedded yet
        job_postings = (
            db_session.query(models.JobPosting)
            .outerjoin(
                models.JobEmbeddingTFIDF,
                models.JobPosting.id == models.JobEmbeddingTFIDF.job_posting_id
            )
            .filter(
                models.JobPosting.desc_tfidf.isnot(None),
                models.JobEmbeddingTFIDF.job_posting_id.is_(None)
            )
            .all()
        )

        if not job_postings:
            print("No job postings found with TF-IDF descriptions that need embedding.")

        else:
            job_descriptions = [job.desc_tfidf for job in job_postings]

            print(f"Embedding {len(job_descriptions)} job descriptions using TF-IDF...")

            # Load the fitted TF-IDF vectorizer
            embedding_service = load_vectorizer("tfidf_vectorizer.pkl")

            # Transform job descriptions and save embeddings to DB
            tfidf_vectors = embedding_service.transform(job_descriptions).toarray()
            
            print("Saving TF-IDF embeddings to database...")
            save_embeddings(job_ids, tfidf_vectors, "TF-IDF", db_session)
        
    except Exception as e:
        print("Exception during TF-IDF transformation:", e)