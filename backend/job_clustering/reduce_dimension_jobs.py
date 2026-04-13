import umap
import numpy as np
from backend.config import EMBEDDING_MODEL
from backend.config import UMAP_PARAMS
import backend.database as database
import backend.models as models

def reduce_dimensions_umap(embeddings):
    reducer = umap.UMAP(**UMAP_PARAMS)
    reduced = reducer.fit_transform(embeddings)
    return reduced

def save_reduced_job_embeddings(embedding_ids: list[int], reduced_embeddings: np.ndarray, model_version: str, reduction_method: str, db_session):
    """Save reduced embeddings directly to database"""

    # Check that table is empty before trying to save
    if db_session.query(models.ReducedEmbedding).first():
        print("ReducedEmbedding table is not empty. Skipping save to avoid duplicates.")
        return
    
    reduced_embeddings = [
        models.ReducedEmbedding(
            reduced_embedding=reduced_embedding.tolist(),
            model_version=model_version,
            job_embedding_id=job_embedding_id,
            reduction_method=reduction_method
        )
        for job_embedding_id, reduced_embedding in zip(embedding_ids, reduced_embeddings)
    ]

    db_session.add_all(reduced_embeddings)  # bulk insert
    db_session.commit()
    
    print(f"Saved {len(embedding_ids)} reduced job embeddings to the database.")

def main():
    SessionLocal = database.SessionLocal
    # Reduce job embeddings with UMAP and save to database
    db_session = SessionLocal()
    try:
        job_embeddings = db_session.query(models.JobEmbeddingSBERT).all()
        
        job_embedding_ids = [je.id for je in job_embeddings]
        job_embeddings_embeddings = [je.embedding for je in job_embeddings]

        print(f"Reducing {len(job_embeddings_embeddings)} job embeddings using UMAP...")
        umap_embeddings = reduce_dimensions_umap(job_embeddings_embeddings)

        print("Saving UMAP-reduced embeddings to database...")
        save_reduced_job_embeddings(job_embedding_ids, umap_embeddings, EMBEDDING_MODEL, "UMAP", db_session)

    except Exception as e:
        print("Exception reducing job embeddings and saving to database:", e)
    finally:
        db_session.close()

if __name__ == "__main__":
  main()