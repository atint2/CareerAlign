import umap
from config import UMAP_PARAMS

def reduce_dimensions(embeddings):
    reducer = umap.UMAP(**UMAP_PARAMS)
    reduced = reducer.fit_transform(embeddings)
    return reduced
