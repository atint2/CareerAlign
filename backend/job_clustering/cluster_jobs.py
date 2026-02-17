import hdbscan 
from config import HDBSCAN_PARAMS 
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
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

def assign_outliers_with_knn(embeddings, labels):
    """
    Replace HDBSCAN outlier labels (-1) with the label of the nearest
    non-outlier embedding.
    """
    embeddings = np.asarray(embeddings)
    labels = np.asarray(labels)

    # Mask for clustered vs outliers
    clustered_mask = labels != -1
    outlier_mask = labels == -1

    # If no outliers or no clusters, return original labels
    if not np.any(outlier_mask) or not np.any(clustered_mask):
        return labels

    # Fit nearest neighbors on clustered points only
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(embeddings[clustered_mask])

    # Find nearest clustered point for each outlier
    distances, indices = nn.kneighbors(embeddings[outlier_mask])

    # Map indices back to cluster labels
    nearest_labels = labels[clustered_mask][indices.flatten()]

    # Replace -1 labels
    new_labels = labels.copy()
    new_labels[outlier_mask] = nearest_labels

    return new_labels

def compute_cluster_centroids(embeddings, labels):
    """
    Returns:
        dict {cluster_id: centroid_vector}
    """
    centroids = {}

    unique_clusters = [c for c in np.unique(labels) if c != -1]

    for cluster_id in unique_clusters:
        cluster_points = embeddings[labels == cluster_id]
        centroids[cluster_id] = cluster_points.mean(axis=0)

    return centroids

def merge_similar_clusters(centroids, labels, threshold=0.95):
    """
    Merge clusters whose centroid cosine similarity exceeds threshold.
    Returns updated labels.
    """
    cluster_ids = sorted(centroids.keys())
    centroid_matrix = np.array([centroids[c] for c in cluster_ids])

    sim_matrix = cosine_similarity(centroid_matrix)

    # Map each cluster to a representative cluster
    parent = {cid: cid for cid in cluster_ids}

    for i in range(len(cluster_ids)):
        for j in range(i + 1, len(cluster_ids)):
            if sim_matrix[i, j] > threshold:
                parent[cluster_ids[j]] = parent[cluster_ids[i]]

    # Relabel
    new_labels = labels.copy()
    for old, new in parent.items():
        new_labels[labels == old] = new

    return new_labels

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
        # Retrieve reduced embeddings along with their job posting IDs 
        rows = ( 
            db_session.query( 
                models.ReducedEmbedding.reduced_embedding, 
                models.JobPosting.id.label("job_posting_id"), 
                models.JobPosting.desc_sbert
            ) 
            .join( 
                models.JobEmbedding, 
                models.JobEmbedding.id == models.ReducedEmbedding.job_embedding_id, 
            ) 
            .join( 
                models.JobPosting, 
                models.JobPosting.id == models.JobEmbedding.job_posting_id, 
            ) 
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
        
        num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        num_outliers = np.sum(cluster_labels == -1)
        print(f"HDBSCAN Clusters: {num_clusters}, Outliers: {num_outliers}")

        # Reassign outliers using nearest neighbors
        cluster_labels = assign_outliers_with_knn(embeddings, cluster_labels)

        centroids = compute_cluster_centroids(embeddings, cluster_labels)
        cluster_labels = merge_similar_clusters(
            centroids,
            cluster_labels,
            threshold=0.999,
        )
        num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        print("Number of clusters after merging similar ones:", num_clusters)

        # Bulk update cluster IDs for job postings 
        for row, label in zip(rows, cluster_labels): 
            db_session.query(models.JobPosting).filter( 
                models.JobPosting.id == row.job_posting_id 
            ).update( 
                {"cluster_id": None if label == -1 else int(label)}, 
                synchronize_session=False, 
            ) 
        db_session.commit() 

    except Exception as e: 
        db_session.rollback() 
        print("Exception during clustering:", e) 
    finally: 
        db_session.close() 

if __name__ == "__main__": 
    main() 