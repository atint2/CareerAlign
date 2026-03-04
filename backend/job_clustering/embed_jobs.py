import numpy as np
import backend.database as database
import backend.models as models
from backend.config import EMBEDDING_MODEL
from backend.services.sbert_embedder import SBERTEmbeddingService
import asyncio

async def save_job_embeddings(job_ids: list[int], embeddings: np.ndarray, model_version: str, db_session):
    # Use endpoint in main.py to save embeddings
    from backend.main import create_job_posting_embedding
    from backend.main import EmbeddingBase
    for job_id, embedding in zip(job_ids, embeddings):
        embedding_obj = EmbeddingBase(
            embedding=embedding.tolist(),
            model_version=model_version,
            job_posting_id=job_id
        )

        await create_job_posting_embedding(embedding_obj, db_session)
    print(f"Saved {len(job_ids)} job embeddings to the database.")

async def main():
    SessionLocal = database.SessionLocal
    db_session = SessionLocal()
    try:
        # Fetch all job postings with SBERT descriptions
        job_postings = db_session.query(models.JobPosting).filter(
            models.JobPosting.desc_sbert.isnot(None)
        ).all()
        
        job_ids = [jp.id for jp in job_postings]
        job_descriptions = [jp.desc_sbert for jp in job_postings]

        print(f"Embedding {len(job_descriptions)} job descriptions...")
        embedding_service = SBERTEmbeddingService()
        embeddings = embedding_service.embed(job_descriptions)

        print("Saving embeddings to database...")
        # await save_job_embeddings(job_ids, embeddings, EMBEDDING_MODEL, db_session)
    except Exception as e:
        print("Exception embedding job postings and saving to database:", e)

if __name__ == "__main__":
  asyncio.run(main())