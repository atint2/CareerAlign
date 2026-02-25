import umap
import numpy as np
from pathlib import Path
import sys

def setup_backend_imports(path="backend"):
	# Ensure backend/ is on sys.path so its modules import as top-level modules
	root = Path(__file__).resolve().parents[2]
	backend_dir = root / path
	sys.path.insert(0, str(backend_dir))

def reduce_dimensions_umap(embeddings, UMAP_PARAMS):
    reducer = umap.UMAP(**UMAP_PARAMS)
    reduced = reducer.fit_transform(embeddings)
    return reduced

def save_reduced_job_embeddings(cluster_embedding_ids: list[int], reduced_embeddings: np.ndarray, model_version: str, reduction_method: str, db_session):
    """Save reduced embeddings directly to database"""
    import models
    
    for cluster_embedding_id, reduced_embedding in zip(cluster_embedding_ids, reduced_embeddings):
        # Ensure reduced_embedding matches Vector(15) size
        if len(reduced_embedding) != 15:
            raise ValueError(f"Expected 15 dimensions, got {len(reduced_embedding)}")
        
        db_reduced_embedding = models.ReducedClusterEmbedding(
            reduced_embedding=reduced_embedding.tolist(),
            model_version=model_version,
            cluster_embedding_id=cluster_embedding_id,
            reduction_method=reduction_method
        )
        db_session.add(db_reduced_embedding)
    
    db_session.commit()
    print(f"Saved {len(cluster_embedding_ids)} reduced cluster embeddings to the database.")

def main():
    setup_backend_imports()
    try:
        import database
        import models
        from config import EMBEDDING_MODEL, UMAP_PARAMS
    except Exception as e:
        print("Exception importing backend modules:", e)
        return

    SessionLocal = database.SessionLocal
    # Reduce cluster embeddings with UMAP and save to database
    db_session = SessionLocal()
    try:
        cluster_embeddings = db_session.query(models.ClusterEmbeddingSBERT).all()
        
        cluster_embedding_ids = [ce.id for ce in cluster_embeddings]
        cluster_embeddings_embeddings = [ce.embedding for ce in cluster_embeddings]

        print(f"Reducing {len(cluster_embeddings_embeddings)} cluster embeddings using UMAP...")
        umap_embeddings = reduce_dimensions_umap(cluster_embeddings_embeddings, UMAP_PARAMS)

        print("Saving UMAP-reduced embeddings to database...")
        save_reduced_job_embeddings(cluster_embedding_ids, umap_embeddings, EMBEDDING_MODEL, "UMAP", db_session)

    except Exception as e:
        print("Exception reducing cluster embeddings and saving to database:", e)

    finally:
        db_session.close()

if __name__ == "__main__":
  main()