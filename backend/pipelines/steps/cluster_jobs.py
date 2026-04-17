from git import db
import hdbscan 
from backend.app.config import HDBSCAN_PARAMS 
import backend.app.database as database
import backend.app.models as models
from collections import Counter
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 

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

def update_cluster_counts(db_session, cluster_labels):
    # Update DB with cluster information
    # Count postings by cluster
    cluster_counts = Counter(
        label for label in cluster_labels if label != -1
    )

    existing_clusters = {
        c.cluster_id: c
        for c in db_session.query(models.Cluster).all()
    }

    try:
        for cid, count in cluster_counts.items():
            if cid in existing_clusters:
                existing_clusters[cid].num_postings = count
            else:
                db_session.add(models.Cluster(
                    cluster_id=int(cid),
                    general_job_desc_raw=None,
                    num_postings=count,
                ))
        
        db_session.commit() 
    except Exception as e:
        db_session.rollback()
        print("Exception updating cluster counts:", e)

def update_posting_clusters(db_session, rows, cluster_labels):
    try:
        updates = [
                {
                    "id": row.job_posting_id,
                    "cluster_id": None if label == -1 else int(label),
                }
                for row, label in zip(rows, cluster_labels)
        ]
        db_session.bulk_update_mappings(models.JobPosting, updates)
        db_session.commit()
    except Exception as e:
        db_session.rollback()
        print("Exception updating job posting clusters:", e)

def run(db_session):
    # Retrieve reduced embeddings from database and cluster them 
    try: 
        # Retrieve current cluster assignments for all job postings
        clusters = db_session.query(models.Cluster).all()
        if clusters:
            print("Clusters already exist in the database. Skipping clustering step.")
            return

        # Retrieve reduced embeddings along with their job posting IDs 
        rows = ( 
            db_session.query( 
                models.ReducedEmbedding.reduced_embedding, 
                models.JobPosting.id.label("job_posting_id"), 
                models.JobPosting.desc_sbert
            ) 
            .join( 
                models.JobEmbeddingSBERT, 
                models.JobEmbeddingSBERT.id == models.ReducedEmbedding.job_embedding_id, 
            ) 
            .join( 
                models.JobPosting, 
                models.JobPosting.id == models.JobEmbeddingSBERT.job_posting_id, 
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

        num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        print("Number of clusters after merging similar ones:", num_clusters)

        # Update cluster counts in the database
        update_cluster_counts(db_session, cluster_labels)

        # Bulk update cluster IDs for job postings 
        update_posting_clusters(db_session, rows, cluster_labels)


    except Exception as e: 
        db_session.rollback() 
        print("Exception during clustering:", e) 
    finally: 
        db_session.close() 