from pathlib import Path
import sys

def setup_backend_imports(path="backend"): 
    # Ensure backend/ is on sys.path so its modules import as top-level modules 
    root = Path(__file__).resolve().parents[2] 
    backend_dir = root / path 
    sys.path.insert(0, str(backend_dir)) 
    
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
        # Retrieve all clusters that are missing content_sbert or content_tfidf
        clusters = db_session.query(models.Cluster).all()

        if not clusters:
            print("No clusters found that require pre-processing.")
            return
    except Exception as e:
        print("Exception querying database for clusters:", e)
        return
        
    print("Starting cluster description preparation for models...")
    try:
        setup_backend_imports("data/scripts")

        from preprocessor_sbert import SBERTPreprocessor
        from preprocessor_tfidf import TFIDFPreprocessor

        sbert_prep = SBERTPreprocessor()
        tfidf_prep = TFIDFPreprocessor()

        for cluster in clusters:
            cluster.general_job_desc_sbert = sbert_prep.clean_text_sbert(cluster.general_job_desc_raw)
            cluster.general_job_desc_tfidf = tfidf_prep.clean_text_tfidf(cluster.general_job_desc_raw)

        db_session.commit()
    except Exception as e:
        print("Exception during cluster description pre-processing:", e)
        db_session.rollback()
        return
    finally:
        print("Finished cluster description preparation.")
        db_session.close()

if __name__ == "__main__":
    main()