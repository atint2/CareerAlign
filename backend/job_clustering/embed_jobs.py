from sentence_transformers import SentenceTransformer
import numpy as np
from config import EMBEDDING_MODEL
from pathlib import Path
import sys
import asyncio

def setup_backend_imports():
	# Ensure backend/ is on sys.path so its modules import as top-level modules
	root = Path(__file__).resolve().parents[2]
	backend_dir = root / "backend"
	sys.path.insert(0, str(backend_dir))

def embed_job_descriptions(job_descriptions: list[str]) -> np.ndarray:
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(job_descriptions, show_progress_bar=True, normalize_embeddings=True)
    return np.array(embeddings)

async def save_job_embeddings(job_ids: list[int], embeddings: np.ndarray, model_version: str, db_session):
    # Use endpoint in main.py to save embeddings
    from main import create_job_embedding
    from main import EmbeddingBase
    for job_id, embedding in zip(job_ids, embeddings):
        embedding_obj = EmbeddingBase(
            embedding=embedding.tolist(),
            model_version=model_version,
            job_posting_id=job_id
        )

        await create_job_embedding(embedding_obj, db_session)
    print(f"Saved {len(job_ids)} job embeddings to the database.")

async def main():

    setup_backend_imports()

    # Import backend modules after adjusting sys.path
    try:
        import database
        import models
    except Exception as e:
        print("Exception importing backend modules:", e)

    SessionLocal = database.SessionLocal
    db_session = SessionLocal()
    try:
        # Fetch all job postings without embeddings
        job_postings = db_session.query(models.JobPosting).filter(
            models.JobPosting.desc_sbert.isnot(None)
        ).all()
        
        job_ids = [jp.id for jp in job_postings]
        job_descriptions = [jp.desc_sbert for jp in job_postings]

        print(f"Embedding {len(job_descriptions)} job descriptions...")
        embeddings = embed_job_descriptions(job_descriptions)

        print("Saving embeddings to database...")
        await save_job_embeddings(job_ids, embeddings, EMBEDDING_MODEL, db_session)
    except Exception as e:
        print("Exception embedding job postings and saving to database:", e)

if __name__ == "__main__":
  asyncio.run(main())