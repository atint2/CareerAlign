from pathlib import Path
import sys
import pickle

def setup_backend_imports(path="backend"):
    # Ensure backend/ is on sys.path so its modules import as top-level modules
    root = Path(__file__).resolve().parents[2]
    backend_dir = root / path
    sys.path.insert(0, str(backend_dir))

def load_vectorizer(path="tfidf_vectorizer.pkl"):
    """
    Load the previously fitted TF-IDF vectorizer from disk.
    """
    with open(path, "rb") as f:
        vectorizer = pickle.load(f)

    # Wrap into the embedding service
    setup_backend_imports("backend/services")
    from tf_idf_embedder import TFIDFEmbeddingService
    embedding_service = TFIDFEmbeddingService()
    embedding_service.vectorizer = vectorizer
    return embedding_service

def main():
    setup_backend_imports()

    try:
        import database
        import models
    except Exception as e:
        print("Exception importing backend modules:", e)
        return

    # Initialize database session
    SessionLocal = database.SessionLocal
    db_session = SessionLocal()

    try:
        # Fetch all job descriptions
        job_descs = db_session.query(models.Cluster).filter(
            models.Cluster.general_job_desc_tfidf != None
        ).all()
    except Exception as e:
        print("Exception during database retrieval:", e)
        db_session.close()
        return

    # Load the fitted TF-IDF vectorizer and transform job descriptions, then save embeddings to DB
    # try:
    #     # Load the fitted TF-IDF vectorizer
    #     embedding_service = load_vectorizer("tfidf_vectorizer.pkl")
    #     print("Loaded TF-IDF vectorizer from disk.")

    #     # Check for existing cluster embeddings to avoid duplicates
    #     existing_cluster_ids = {
    #         e.cluster_id
    #         for e in db_session.query(models.ClusterEmbeddingTFIDF.cluster_id).all()
    #     }

    #     # Transform job descriptions and save embeddings to DB
    #     for cluster in job_descs:
    #         if cluster.id in existing_cluster_ids:
    #             continue  # Skip if embedding already exists

    #         tfidf_vector = embedding_service.transform([cluster.general_job_desc_tfidf])
    #         embedding_vector = tfidf_vector.toarray()[0].tolist()

    #         embedding_obj = models.ClusterEmbeddingTFIDF(
    #             embedding=embedding_vector,
    #             cluster_id=cluster.id
    #         )
    #         db_session.add(embedding_obj)

    #     db_session.commit()
    #     print(f"Inserted/updated {len(job_descs)} cluster embeddings.")

    # except Exception as e:
    #     print("Exception during TF-IDF transformation or DB insertion:", e)
    
    # Embed job descriptions using SBERT and save to DB
    try:
        setup_backend_imports("backend/services")
        from sbert_embedder import SBERTEmbeddingService
        embedding_service = SBERTEmbeddingService()

        # Check for existing cluster embeddings to avoid duplicates
        existing_cluster_ids = {
            e.cluster_id
            for e in db_session.query(models.ClusterEmbeddingSBERT.cluster_id).all()
        }

        for cluster in job_descs:
            if cluster.id in existing_cluster_ids:
                continue  # Skip if embedding already exists

            sbert_embedding = embedding_service.embed([cluster.general_job_desc_sbert])[0].tolist()

            embedding_obj = models.ClusterEmbeddingSBERT(
                embedding=sbert_embedding,
                cluster_id=cluster.id
            )
            db_session.add(embedding_obj)

        db_session.commit()
        print(f"Inserted/updated {len(job_descs)} SBERT cluster embeddings.")

    except Exception as e:
        print("Exception during SBERT embedding or DB insertion:", e)

    finally:
        db_session.close()

if __name__ == "__main__":
    main()