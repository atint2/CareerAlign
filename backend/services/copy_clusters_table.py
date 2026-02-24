"""
Script to copy data from clusters table to clusters_copy table, which is used for testing new embedding methods without affecting the original data. This allows us to experiment with different embedding techniques and compare results while keeping the original clusters intact.
"""

import re
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
        
    # Initialize database session
    SessionLocal = database.SessionLocal
    session = SessionLocal()
    try:
        # Fetch all clusters from the original table
        clusters = session.query(models.Cluster).filter(models.Cluster.cluster_desc != None).all()

        # Prepare ClusterCopy objects for bulk insertion
        cluster_copies = []
        for c in clusters:
            # Extract title from cluster_desc using regex
            title_match = re.search(
                r"\*\*Role Title:\*\*\s*(.+)",
                c.cluster_desc
            )

            if title_match:
                title = title_match.group(1).strip()

            # Extract general job description from cluster_desc using regex
            desc_match = re.search(
                r"\*\*Professional Summary:\*\*\s*(.+)",
                c.cluster_desc,
                re.DOTALL
            )

            if desc_match:
                description = desc_match.group(1).strip()
            else:
                print(f"Warning: Could not extract description for cluster_id {c.cluster_id}")

            # Create ClusterCopy object with extracted information
            copy = models.ClusterCopy(
                cluster_id=c.cluster_id,
                title=title,
                general_job_desc_raw=description,
                num_postings=c.num_postings
            )
            cluster_copies.append(copy)

        # Bulk insert into clusters_copy table
        if cluster_copies:
            session.bulk_save_objects(cluster_copies)
            session.commit()
            print(f"Copied {len(cluster_copies)} clusters to clusters_copy table.")
        else:
            print("No clusters found to copy.")

    finally:
        session.close()

if __name__ == "__main__":
    main()
