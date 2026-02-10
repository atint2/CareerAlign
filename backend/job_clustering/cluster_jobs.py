import hdbscan
from config import HDBSCAN_PARAMS

def cluster_jobs_hdbscan(reduced_embeddings):
    # Cluster the reduced embeddings using HDBSCAN
    clusterer = hdbscan.HDBSCAN()