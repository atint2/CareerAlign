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
        # Retrieve all resumes that are missing content_sbert or content_tfidf
        resumes = db_session.query(models.Resume).filter(
            (models.Resume.content_sbert == None) | (models.Resume.content_tfidf == None)
        ).all()

        if not resumes:
            print("No resumes found that require pre-processing.")
            return
    except Exception as e:
        print("Exception querying database for resumes:", e)
        return
        
    print("Starting resume preparation for models...")
    try:
        setup_backend_imports("data/scripts")

        from preprocessor_sbert import SBERTPreprocessor
        from preprocessor_tfidf import TFIDFPreprocessor

        sbert_prep = SBERTPreprocessor()
        tfidf_prep = TFIDFPreprocessor()

        for resume in resumes:
            if resume.content_sbert is None:
                resume.content_sbert = sbert_prep.clean_text_sbert(resume.content)
            if resume.content_tfidf is None:
                resume.content_tfidf = tfidf_prep.clean_text_tfidf(resume.content)

        db_session.commit()
    except Exception as e:
        print("Exception during resume pre-processing:", e)
        db_session.rollback()
        return
    finally:
        print("Finished resume preparation.")
        db_session.close()

if __name__ == "__main__":
    main()