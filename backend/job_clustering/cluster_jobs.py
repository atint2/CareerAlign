import hdbscan 
from config import HDBSCAN_PARAMS 
import numpy as np 
from pathlib import Path 
import sys 

def setup_backend_imports(): 
    # Ensure backend/ is on sys.path so its modules import as top-level modules 
    root = Path(__file__).resolve().parents[2] 
    backend_dir = root / "backend" 
    sys.path.insert(0, str(backend_dir)) 
 
def cluster_jobs_hdbscan(reduced_embeddings): 
    if len(reduced_embeddings) == 0: 
        return np.array([]) 

    # Cluster the reduced embeddings using HDBSCAN 
    clusterer = hdbscan.HDBSCAN(**HDBSCAN_PARAMS) 
    clusterer.fit(reduced_embeddings) 
    return clusterer.labels_  # Return the cluster labels assigned to each job 

def main(): 
    setup_backend_imports() 
    try: 
        import database 
        import models 
    except Exception as e: 
        print("Exception importing backend modules:", e) 
        return 

    SessionLocal = database.SessionLocal 
    # Retrieve reduced embeddings from database and cluster them 
    db_session = SessionLocal() 
    try: 
        # Retrieve UMAP reduced embeddings along with their job posting IDs 
        rows = ( 
            db_session.query( 
                models.ReducedEmbedding.reduced_embedding, # Only retrieve umap embeddings 
                models.JobPosting.id.label("job_posting_id"), 
            ) 
            .join( 
                models.JobEmbedding, 
                models.JobEmbedding.id == models.ReducedEmbedding.job_embedding_id, 
            ) 
            .join( 
                models.JobPosting, 
                models.JobPosting.id == models.JobEmbedding.job_posting_id, 
            ) 
            .filter(models.ReducedEmbedding.reduction_method == "UMAP") 
            .order_by(models.ReducedEmbedding.job_embedding_id) 
            .all() 
        ) 
        if not rows: 
            print("No embeddings found. Nothing to cluster.") 
            return 

        # Convert embeddings to numpy array 
        embeddings = np.array([np.array(r.reduced_embedding, dtype=float) for r in rows]) 

        # Run clustering 
        cluster_labels = cluster_jobs_hdbscan(embeddings) 

        # Compare cluster labels and evaluate which reduction method leads to better clustering (e.g. more clusters, fewer outliers) 
        num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0) 
        num_outliers = np.sum(cluster_labels == -1) 
        print(f"UMAP Clusters: {num_clusters}, UMAP Outliers: {num_outliers}") 

        # # Bulk update cluster IDs for job postings 
        # for row, label in zip(rows, cluster_labels): 
        #     db_session.query(models.JobPosting).filter( 
        #         models.JobPosting.id == row.job_posting_id 
        #     ).update( 
        #         {"cluster_id": None if label == -1 else int(label)}, 
        #         synchronize_session=False, 
        #     ) 
        # db_session.commit() 

    except Exception as e: 
        db_session.rollback() 
        print("Exception during clustering:", e) 
    finally: 
        db_session.close() 

if __name__ == "__main__": 
    main() 