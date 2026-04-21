
"""
Save processed job postings CSV into the database's `job_postings` table.
The script will skip rows whose `job_id` already exist in the database to avoid duplicates.
"""
from pathlib import Path
import sys
from dotenv import load_dotenv

from backend.app import database
load_dotenv() 
from tqdm import tqdm
tqdm.pandas()
import pandas as pd
from backend.app import models

def save_job_postings_to_db(dataset_filepath):
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
		# Filter existing ids to avoid duplicates
		existing = set(
			str(r[0]) for r in session.query(models.JobPosting.job_id).all()
		)

		df_new = df[~df["job_id"].astype(str).isin(existing)].copy()

		print(f"Found {len(df)} total rows.")
		print(f"{len(df_new)} new rows to process.")

		if df_new.empty:
			print("No new postings to insert.")
			return

		# Initialize preprocessors
		sys.path.insert(0, str(Path(__file__).parent))
		from preprocessor_sbert import SBERTPreprocessor
		from preprocessor_tfidf import TFIDFPreprocessor

		sbert_prep = SBERTPreprocessor()
		tfidf_prep = TFIDFPreprocessor()

		# Batch preprocess descriptions and build JobPosting objects
		descriptions = df_new["description"].fillna("").astype(str)

		print("Cleaning TF-IDF text...")
		df_new["desc_tfidf"] = descriptions.progress_apply(
			tfidf_prep.clean_text_tfidf
		)

		print("Cleaning SBERT text...")
		df_new["desc_sbert"] = descriptions.progress_apply(
			sbert_prep.clean_text_sbert
		)

		# Insert into database
		to_insert = []

		for _, row in tqdm(
			df_new.iterrows(),
			total=len(df_new),
			desc="Building DB objects"
		):
			jp = models.JobPosting(
				job_id=row["job_id"],
				title=row.get("title"),
				desc_raw=row.get("description"),
				desc_sbert=row.get("desc_sbert"),
				desc_tfidf=row.get("desc_tfidf"),
				formatted_work_type=row.get("formatted_work_type"),
				company=row.get("company_name"),
				formatted_experience_level=row.get("formatted_experience_level"),
				cluster_id=row.get("cluster_id")
			)
			to_insert.append(jp)

		print("Inserting into database...")
		session.add_all(to_insert)
		session.commit()

		print(f"Inserted {len(to_insert)} new job_postings.")

	finally:
		session.close()

def save_resumes_to_db(dataset_filepath):
	# Read CSV
	df = pd.read_csv(dataset_filepath)
	if df.empty:
		print("CSV is empty — nothing to do.")
		return
	
	# Ensure at least required columns exist
	required = ["ID", "Resume_str"]
	missing = [c for c in required if c not in df.columns]
	if missing:
		raise SystemExit(f"CSV is missing required columns: {missing}")
	
	# Normalize NaNs to None
	df = df.where(pd.notnull(df), None)

	# Initialize database session
	SessionLocal = database.SessionLocal
	session = SessionLocal()
	try:
		# Filter existing ids to avoid duplicates
		existing = set(
			r[0] for r in session.query(models.Resume.resume_id).all()
		)

		df_new = df[~df["ID"].isin(existing)].copy()

		print(f"Found {len(df)} total rows.")
		print(f"{len(df_new)} new rows to process.")

		if df_new.empty:
			print("No new resumes to insert.")
			return
		
		# Initialize preprocessors
		sys.path.insert(0, str(Path(__file__).parent))
		from preprocessor_sbert import SBERTPreprocessor
		from preprocessor_tfidf import TFIDFPreprocessor

		sbert_prep = SBERTPreprocessor()
		tfidf_prep = TFIDFPreprocessor()

		# Batch preprocess resumes and build Resume objects
		resume_texts = df_new["Resume_str"].fillna("").astype(str)

		print("Cleaning TF-IDF text...")
		df_new["content_tfidf"] = resume_texts.progress_apply(
			tfidf_prep.clean_text_tfidf
		)

		print("Encoding SBERT embeddings...")
		df_new["content_sbert"] = resume_texts.progress_apply(
			sbert_prep.clean_text_sbert
		)

		# Insert into database
		to_insert = []

		for _, row in df_new.iterrows():
			rid = str(row["ID"]).strip()
			if rid in existing:
				continue
			r = models.Resume(
				resume_id=rid,
				content_raw=row.get("Resume_str"),
				content_sbert=row.get("content_sbert"),
				content_tfidf=row.get("content_tfidf"),
			)
			to_insert.append(r)

		print(f"Found {len(df)} rows in CSV; {len(to_insert)} new resumes to insert.")

        # Bulk insert new resumes
		if to_insert:
			session.add_all(to_insert)
			session.commit()
			print(f"Inserted {len(to_insert)} new resumes.")
		else:
			print("No new resumes to insert.")

	finally:
		session.close()

def save_skills_to_db(dataset_filepath):
	# Read CSV
	df = pd.read_csv(dataset_filepath)
	if df.empty:
		print("CSV is empty — nothing to do.")
		return
	
	# Ensure at least required columns exist
	required = ["Example"]
	missing = [c for c in required if c not in df.columns]
	if missing:
		raise SystemExit(f"CSV is missing required columns: {missing}")
	
	# Normalize NaNs to None
	df = df.where(pd.notnull(df), None)

	# Initialize database session
	SessionLocal = database.SessionLocal
	session = SessionLocal()
	try:
		# Filter existing skills to avoid duplicates
		existing = set(
			r[0] for r in session.query(models.Skill.skill).all()
		)

		df_new = df[~df["Example"].isin(existing)].copy()

		print(f"Found {len(df)} total rows.")
		print(f"{len(df_new)} new rows to process.")

		if df_new.empty:
			print("No new skills to insert.")
			return

		# Insert into database
		to_insert = []

		for _, row in df_new.iterrows():
			skill = str(row["Example"]).strip()
			if skill in existing:
				continue
			r = models.Skill(
				skill=skill,
				hot_technology=row.get("Hot Technology"),
				in_demand=row.get("In Demand")
			)
			to_insert.append(r)

		print(f"Found {len(df)} rows in CSV; {len(to_insert)} new skills to insert.")

        # Bulk insert new skills
		if to_insert:
			session.add_all(to_insert)
			session.commit()
			print(f"Inserted {len(to_insert)} new skills.")
		else:
			print("No new skills to insert.")

	finally:
		session.close()
	
def main():
	resume_dataset = Path(__file__).resolve().parents[1] / "processed" / "cleaned_resumes.csv"
	job_posting_dataset = Path(__file__).resolve().parents[1] / "processed" / "cleaned_job_postings.csv"
	skills_dataset = Path(__file__).resolve().parents[1] / "processed" / "cleaned_skills.csv"

	# Save datasets to DB
	save_job_postings_to_db(job_posting_dataset)
	save_resumes_to_db(resume_dataset)
	save_skills_to_db(skills_dataset)

if __name__ == "__main__":
	main()

