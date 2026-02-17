from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
from pathlib import Path 
import sys 

def setup_backend_imports(): 
    # Ensure backend/ is on sys.path so its modules import as top-level modules 
    root = Path(__file__).resolve().parents[2] 
    backend_dir = root / "backend" 
    sys.path.insert(0, str(backend_dir)) 

def compute_cluster_keywords(texts, labels, top_k=20):
    """
    Returns:
        dict {cluster_id: [top keywords]}
    """

    custom_stop_words = list(
        text.ENGLISH_STOP_WORDS.union({"occasionally", "usually"})
    )

    vectorizer = TfidfVectorizer(
        stop_words=custom_stop_words,
        max_df=0.8,
        min_df=5
    )

    tfidf_matrix = vectorizer.fit_transform(texts)
    terms = np.array(vectorizer.get_feature_names_out())

    cluster_keywords = {}

    for cluster_id in np.unique(labels):
        if cluster_id is None:
            continue

        cluster_docs = tfidf_matrix[labels == cluster_id]
        mean_scores = np.asarray(cluster_docs.mean(axis=0)).ravel()

        top_indices = mean_scores.argsort()[::-1][:top_k]
        cluster_keywords[cluster_id] = terms[top_indices].tolist()

    return cluster_keywords

def centroid_label(embeddings, titles):
    # Label clusters by finding job title which is closest to centroid
    centroid = embeddings.mean(axis=0, keepdims=True)
    sims = cosine_similarity(embeddings, centroid).ravel()
    return titles[np.argmax(sims)]

def main():
    setup_backend_imports()

    try:
        import database
        import models
    except Exception as e:
        print("Exception importing backend modules:", e)
        return

    SessionLocal = database.SessionLocal
    db_session = SessionLocal()

    try:
        # Retrieve descriptions + cluster IDs
        rows = (
            db_session.query(
                models.JobPosting.desc_sbert,
                models.JobPosting.cluster_id,
            )
            .filter(models.JobPosting.cluster_id != None)
            .all()
        )

        if not rows:
            print("No clustered job postings found.")
            return

        texts = [r.desc_sbert or "" for r in rows]
        labels = np.array([r.cluster_id for r in rows])

        keywords = compute_cluster_keywords(texts, labels, top_k=20)

        # Print first five clusters
        for cid in sorted(keywords.keys())[:5]:
            print(f"\nCluster {cid}")
            print("Top keywords:", ", ".join(keywords[cid]))

    except Exception as e:
        print("Exception during keyword computation:", e)
    finally:
        db_session.close()

if __name__ == "__main__":
    main()