import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from config import EMBEDDING_MODEL
from config import UMAP_PARAMS
from config import PCA_PARAMS 
from pathlib import Path
import sys

def setup_backend_imports():
	# Ensure backend/ is on sys.path so its modules import as top-level modules
	root = Path(__file__).resolve().parents[2]
	backend_dir = root / "backend"
	sys.path.insert(0, str(backend_dir))

def reduce_dimensions_umap(embeddings):
    reducer = umap.UMAP(**UMAP_PARAMS)
    reduced = reducer.fit_transform(embeddings)
    return reduced

def reduce_dimensions_pca(embeddings):
    # Standardize embeddings before PCA
    scaler = StandardScaler()
    standardized_embeddings = scaler.fit_transform(embeddings)
    reducer = PCA(**PCA_PARAMS)
    reduced = reducer.fit_transform(standardized_embeddings)
    return reduced

def save_reduced_job_embeddings(embedding_ids: list[int], reduced_embeddings: np.ndarray, model_version: str, reduction_method: str, db_session):
    """Save reduced embeddings directly to database"""
    import models

    # Check that table exists before trying to save
    if not db_session.query(models.ReducedEmbedding).first():
        print("ReducedEmbedding table does not exist. Cannot save reduced embeddings.")
        return
    # Check that table is empty before trying to save
    if db_session.query(models.ReducedEmbedding).first():
        print("ReducedEmbedding table is not empty. Skipping save to avoid duplicates.")
        return
    
    for job_embedding_id, reduced_embedding in zip(embedding_ids, reduced_embeddings):
        # Ensure reduced_embedding matches Vector(15) size
        if len(reduced_embedding) != 15:
            raise ValueError(f"Expected 15 dimensions, got {len(reduced_embedding)}")
        
        db_reduced_embedding = models.ReducedEmbedding(
            reduced_embedding=reduced_embedding.tolist(),
            model_version=model_version,
            job_embedding_id=job_embedding_id,
            reduction_method=reduction_method
        )
        db_session.add(db_reduced_embedding)
    
    db_session.commit()
    print(f"Saved {len(embedding_ids)} reduced job embeddings to the database.")

def main():
    setup_backend_imports()
    try:
        import database
        import models
    except Exception as e:
        print("Exception importing backend modules:", e)
        return

    SessionLocal = database.SessionLocal
    # Reduce job embeddings with UMAP and save to database
    db_session = SessionLocal()
    try:
        job_embeddings = db_session.query(models.JobEmbedding).all()
        
        job_embedding_ids = [je.id for je in job_embeddings]
        job_embeddings_embeddings = [je.embedding for je in job_embeddings]

        print(f"Reducing {len(job_embeddings_embeddings)} job embeddings using UMAP...")
        umap_embeddings = reduce_dimensions_umap(job_embeddings_embeddings)

        print(f"Reducing {len(job_embeddings_embeddings)} job embeddings using PCA...")
        pca_embeddings = reduce_dimensions_pca(job_embeddings_embeddings)

        print("Saving UMAP-reduced embeddings to database...")
        save_reduced_job_embeddings(job_embedding_ids, umap_embeddings, EMBEDDING_MODEL, "UMAP", db_session)
        # print("Saving PCA-reduced embeddings to database...")
        # save_reduced_job_embeddings(job_embedding_ids, pca_embeddings, EMBEDDING_MODEL, "PCA", db_session)

    except Exception as e:
        print("Exception reducing job embeddings and saving to database:", e)
    finally:
        db_session.close()

if __name__ == "__main__":
  main()