
"""
Save processed job postings CSV into the database's `job_postings` table.
The script will skip rows whose `job_id` already exist in the database to avoid duplicates.
"""
from pathlib import Path
import sys
from dotenv import load_dotenv
import pandas as pd

def setup_backend_imports():
	# Ensure backend/ is on sys.path so its modules import as top-level modules
	root = Path(__file__).resolve().parents[2]
	backend_dir = root / "backend"
	sys.path.insert(0, str(backend_dir))

def clean_description(text):
	"""Use preprocessors to clean job description text."""
	sys.path.insert(0, str(Path(__file__).parent))
	from preprocessor_sbert import SBERTPreprocessor
	from preprocessor_tfidf import TFIDFPreprocessor

	sbert_prep = SBERTPreprocessor()
	sbert_cleaned = sbert_prep.clean_text_sbert(text)
	tfidf_prep = TFIDFPreprocessor()
	tfidf_cleaned = tfidf_prep.clean_text_tfidf(text)

	return sbert_cleaned, tfidf_cleaned

def main():
	dataset_filepath = Path(__file__).resolve().parents[1] / "processed" / "cleaned_job_postings.csv"

	load_dotenv()  # ensure DATABASE_URL is present for backend/database.py
	setup_backend_imports()

	# Import backend modules after adjusting sys.path
	try:
		import database
		import models
	except Exception as e:
		print("Exception importing backend modules:", e)

	# Read CSV
	df = pd.read_csv(dataset_filepath)
	if df.empty:
		print("CSV is empty — nothing to do.")
		return

	# Ensure at least required columns exist
	required = ["job_id", "title", "description", "formatted_work_type"]
	missing = [c for c in required if c not in df.columns]
	if missing:
		raise SystemExit(f"CSV is missing required columns: {missing}")

	# Normalize NaNs to None
	df = df.where(pd.notnull(df), None)

	SessionLocal = database.SessionLocal
	session = SessionLocal()
	try:
		# Fetch existing job_ids from database to avoid duplicates
		existing = set(r[0] for r in session.query(models.JobPosting.job_id).all())

		to_insert = []
		for _, row in df.iterrows():
			jid = str(row["job_id"]).strip()
			if jid in existing:
				continue
			# Clean description text
			desc_sbert, desc_tfidf = clean_description(row.get("description", ""))
			row["desc_sbert"] = desc_sbert
			row["desc_tfidf"] = desc_tfidf
			jp = models.JobPosting(
				job_id=jid,
				title=row.get("title"),
				desc_raw=row.get("description"),
				desc_sbert=row.get("desc_sbert"),
				desc_tfidf=row.get("desc_tfidf"),
				formatted_work_type=row.get("formatted_work_type"),
				company=row.get("company_name"),
				formatted_experience_level=row.get("formatted_experience_level"),
			)
			to_insert.append(jp)

		print(f"Found {len(df)} rows in CSV; {len(to_insert)} new postings to insert.")

        # Bulk insert new postings
		if to_insert:
			session.add_all(to_insert)
			session.commit()
			print(f"Inserted {len(to_insert)} new job_postings.")
		else:
			print("No new postings to insert.")

	finally:
		session.close()


if __name__ == "__main__":
	main()

