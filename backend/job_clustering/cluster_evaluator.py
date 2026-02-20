import numpy as np 
from collections import Counter
from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path 
import sys 

def setup_backend_imports(): 
    # Ensure backend/ is on sys.path so its modules import as top-level modules 
    root = Path(__file__).resolve().parents[2] 
    backend_dir = root / "backend" 
    sys.path.insert(0, str(backend_dir)) 

def mean_intra_cluster_similarity(embeddings, labels):
    """
    Computes the average cosine similarity between all pairs of points
    within the same cluster.
    """
    unique_clusters = [c for c in np.unique(labels)]
    similarities = []

    for cid in unique_clusters:
        cluster_points = embeddings[labels == cid]

        # Need at least 2 points to compute similarity
        if len(cluster_points) < 2:
            continue

        sim_matrix = cosine_similarity(cluster_points)

        # Take upper triangle without diagonal
        triu_indices = np.triu_indices_from(sim_matrix, k=1)
        similarities.extend(sim_matrix[triu_indices])

    if not similarities:
        return float("nan")

    return float(np.mean(similarities))

def main():
    setup_backend_imports()
    try:
        import database
        import models
    except Exception as e:
        print("Exception importing backend modules:", e)
        return
    
    SessionLocal = database.SessionLocal
    # Retrieve job postings and their cluster assignments from database and evaluate cluster quality
    db_session = SessionLocal()
    try:
        # Retrieve job postings with their cluster IDs
        rows = (
            db_session.query(
                models.ReducedEmbedding.reduced_embedding,
                models.JobPosting.cluster_id,
            )
            .join(
                models.JobEmbedding,
                models.JobEmbedding.id == models.ReducedEmbedding.job_embedding_id,
            )
            .join(
                models.JobPosting,
                models.JobPosting.id == models.JobEmbedding.job_posting_id,
            )
            .filter(models.JobPosting.cluster_id != None)
            .all()
        )

        if not rows:
            print("No clustered job postings found. Nothing to evaluate.")
            return

        embeddings = np.array([np.array(r.reduced_embedding, dtype=float) for r in rows])
        labels = np.array([r.cluster_id for r in rows])

        print(f"Retrieved {len(labels)} clustered job postings for evaluation.")

        # Evaluate cluster quality by printing statistical information about clusters
        cluster_counts = Counter(labels)
        sizes = np.array(list(cluster_counts.values()))
        cluster_median_size = np.median(sizes)
        cluster_mean_size = np.mean(sizes)
        cluster_min_size = min(sizes)
        cluster_max_size = max(sizes)
        cluster_std_size = np.std(sizes)
        print(f"Cluster count: {len(cluster_counts)}")
        print(f"Cluster size statistics: median={cluster_median_size}, mean={cluster_mean_size:.2f}, min={cluster_min_size}, max={cluster_max_size}, std={cluster_std_size:.2f}")

        # Use evaluation metrics like DBI and CHI
        if len(cluster_counts) < 2:
            print("\nNot enough clusters to compute DBI / Silhouette / CHI.")
            return
        
        dbi = davies_bouldin_score(embeddings, labels)
        silhouette = silhouette_score(embeddings, labels, metric="cosine")
        chi = calinski_harabasz_score(embeddings, labels)
        intra_sim = mean_intra_cluster_similarity(embeddings, labels)

        print("\n=== Clustering Quality Metrics ===")
        print(f"Davies-Bouldin Index: {dbi:.4f}")
        print(f"Silhouette Score: {silhouette:.4f}")
        print(f"Calinski-Harabasz Index: {chi:.2f}")
        print(f"Mean Intra-cluster Cosine Similarity: {intra_sim:.4f}")

    except Exception as e:
        print("Exception during cluster evaluation:", e)
    finally:
        db_session.close()

if __name__ == "__main__":
    main()