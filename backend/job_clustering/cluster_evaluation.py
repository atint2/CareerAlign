import numpy as np 
from collections import Counter
from pathlib import Path 
import sys 

def setup_backend_imports(): 
    # Ensure backend/ is on sys.path so its modules import as top-level modules 
    root = Path(__file__).resolve().parents[2] 
    backend_dir = root / "backend" 
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
    # Retrieve job postings and their cluster assignments from database and evaluate cluster quality
    db_session = SessionLocal()
    try:
        # Retrieve job postings with their cluster IDs
        postings = db_session.query(models.JobPosting).filter(models.JobPosting.cluster_id != None).all()
        if not postings:
            print("No clustered job postings found. Nothing to evaluate.")
            return
        cluster_ids = [p.cluster_id for p in postings]
        print(f"Retrieved {len(postings)} clustered job postings for evaluation.")

        # Evaluate cluster quality by printing statistical information about clusters
        cluster_counts = Counter(cluster_ids)
        cluster_median_size = np.median(list(cluster_counts.values()))
        cluster_mean_size = np.mean(list(cluster_counts.values()))
        cluster_min_size = min(cluster_counts.values())
        cluster_max_size = max(cluster_counts.values())
        cluster_std_size = np.std(list(cluster_counts.values()))
        print(f"Cluster count: {len(cluster_counts)}")
        print(f"Cluster size statistics: median={cluster_median_size}, mean={cluster_mean_size:.2f}, min={cluster_min_size}, max={cluster_max_size}, std={cluster_std_size:.2f}")

    except Exception as e:
        print("Exception during cluster evaluation:", e)
    finally:
        db_session.close()

if __name__ == "__main__":
    main()