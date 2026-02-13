from sklearn.manifold import trustworthiness as calc_trustworthiness
import numpy as np
from pathlib import Path
import sys

def setup_backend_imports():
	# Ensure backend/ is on sys.path so its modules import as top-level modules
	root = Path(__file__).resolve().parents[2]
	backend_dir = root / "backend"
	sys.path.insert(0, str(backend_dir))

def evaluate_reduction_quality(original_embeddings, reduced_embeddings, method_name, sample_size=5000):
    """
    Evaluate on a random sample to avoid memory issues
    """
    # Convert to numpy arrays
    original_embeddings = np.array(original_embeddings)
    reduced_embeddings = np.array(reduced_embeddings)

    if len(original_embeddings) > sample_size:
        print(f"Sampling {sample_size} embeddings for trustworthiness calculation...")
        indices = np.random.choice(len(original_embeddings), sample_size, replace=False)
        original_sample = original_embeddings[indices]
        reduced_sample = reduced_embeddings[indices]
    else:
        original_sample = original_embeddings
        reduced_sample = reduced_embeddings
    
    trust_score = calc_trustworthiness(original_sample, reduced_sample)
    print(f"Trustworthiness of {method_name}: {trust_score:.4f} (evaluated on {len(original_sample)} samples)")

def main():
    setup_backend_imports()
    try:
        import database
        import models
    except Exception as e:
        print("Exception importing backend modules:", e)
        return
    
    SessionLocal = database.SessionLocal
    # Retrieve reduced embeddings from database and evaluate quality
    db_session = SessionLocal()
    try:
        # First retrieve id and embedding for all job embeddings in database (nonreduced)
        job_embeddings = db_session.query(models.JobEmbedding).all()
        job_embedding_ids = [je.id for je in job_embeddings]
        job_embeddings_embeddings = [je.embedding for je in job_embeddings]

        # Retrieve reduced embeddings for UMAP, ordered by job_embedding_id
        umap_reduced = db_session.query(models.ReducedEmbedding)\
            .filter(models.ReducedEmbedding.reduction_method == "UMAP")\
            .order_by(models.ReducedEmbedding.job_embedding_id)\
            .all()
        print(f"Retrieved {len(umap_reduced)} UMAP reduced embeddings")
        

        # Create dictionaries for alignment
        umap_dict = {ur.job_embedding_id: ur.reduced_embedding for ur in umap_reduced}

        # Align reduced embeddings with original embeddings
        umap_aligned = []
        original_aligned = []
        
        for je_id, orig_emb in zip(job_embedding_ids, job_embeddings_embeddings):
            if je_id in umap_dict:
                original_aligned.append(orig_emb)
                umap_aligned.append(umap_dict[je_id])
        
        original_aligned = np.array(original_aligned)
        umap_aligned = np.array(umap_aligned)
        
        print(f"\nAligned {len(original_aligned)} embeddings for evaluation")

        # Evaluate reduction quality of each method
        evaluate_reduction_quality(original_aligned, umap_aligned, "UMAP", sample_size=10000)
    except Exception as e:
        print("Exception evaluating reduction quality:", e)
    finally:
        db_session.close()

if __name__ == "__main__":
    main()