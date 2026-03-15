# SBERT Model for generating job embeddings (clustering)
EMBEDDING_MODEL = "all-minilm-l6-v2"   

# For UMAP dimensionality reduction
UMAP_PARAMS = {
    "n_neighbors": 15, 
    "n_components": 15,
    "min_dist": 0.1,
    "metric": "cosine",
    "random_state": 42,
    "verbose": True, 
    "tqdm_kwds": {'colour': 'green'}
}

# For HDBSCAN clustering
HDBSCAN_PARAMS = {
    "min_cluster_size": 40,
    "min_samples": 10,
    "metric": "euclidean",
    "cluster_selection_method": "eom",
}

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
# Define custom stopwords for TF-IDF vectorizer
CUSTOM_STOPWORDS = ENGLISH_STOP_WORDS | {
    "new", "work", "working", "using", "use", "used",
    "experience", "ability", "strong", "good", "knowledge",
    "team", "within", "across", "including", "related",
    "role", "position", "job", "company", "opportunity",
    "responsibility", "requires", "contribute", "seeking", "based",
    "demonstrate", "demonstrating", "demonstrates", "valued",
    "values", "value", "functioning", "ideal"
}