# SBERT Model for generating job embeddings (clustering)
EMBEDDING_MODEL = "all-minilm-l6-v2"    

# For UMAP dimensionality reduction
UMAP_PARAMS = {
    "n_neighbors": 15, 
    "n_components": 50,
    "min_dist": 0.1,
    "metric": "cosine",
    "random_state": 42
}

# For HDBSCAN clustering
HDBSCAN_PARAMS = {
    "min_cluster_size": 15,
    "metric": "euclidean",
    "cluster_selection_method": "eom",
}