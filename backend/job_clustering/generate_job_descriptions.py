from google import genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import numpy as np 
from pathlib import Path 
import sys 
import time

# Load API key
from dotenv import load_dotenv
import os
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

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

def create_llm_prompt(keywords, sample_titles, sample_descriptions):
    """
    Create the prompt that asks LLM to generate a general job description
    Use extracted keywords, sample titles, and sample descriptions
    """
    # Format keywords as a readable list
    keywords_text = ", ".join(keywords) if keywords else "Not specified"
    
    titles_text = "\n".join(f"- {t}" for t in sample_titles[:5]) or "None provided"
    
    desc_text = "\n\n".join(sample_descriptions[:3]) or "None provided"



    prompt = f"""You are generating a concise, generalized job description for a group of similar job postings.
    Use the provided keywords, example job titles, and descriptions to infer:
    - the common role
    - primary responsibilities
    - required skills

    Write:
    - 1 short role title
    - 3–5 sentence professional summary
    - Avoid mentioning clusters, data, or analysis.

    KEYWORDS:
    {keywords_text}

    SAMPLE JOB TITLES:
    {titles_text}

    SAMPLE DESCRIPTIONS:
    {desc_text}
    """

    return prompt.strip()

def generate_job_description(keywords, sample_titles, sample_descriptions):
    """
    Call LLM to generate a generalized job description.
    """
    if not sample_titles and not sample_descriptions:
        return "Insufficient data to generate description."

    try:
        prompt = create_llm_prompt(keywords, sample_titles, sample_descriptions)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text.strip()
    except Exception as e:
        print("LLM generation failed:", e)
        return "Description unavailable."

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
                models.JobPosting.title,
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

        cluster_map = defaultdict(list)

        for r in rows:
            cluster_map[r.cluster_id].append(r)

        # Generate descriptions for clusters
        for cid in sorted(keywords.keys()):
            cluster_rows = cluster_map[cid]

            sample_titles = [r.title for r in cluster_rows if r.title]
            sample_descs = [r.desc_sbert for r in cluster_rows if r.desc_sbert]

            existing = (
                db_session.query(models.Cluster)
                .filter(models.Cluster.cluster_id == int(cid))
                .one_or_none()
            )
            if existing:
                if existing.cluster_desc:
                    pass
                else:
                    # Generate description for cluster
                    description = generate_job_description(
                        keywords[cid],
                        sample_titles,
                        sample_descs,
                    )
                    if description != "Description unavailable.":
                        # Save description to database
                        existing.cluster_desc = description
                        db_session.commit()
                    # Sleep before next iteration
                    time.sleep(30)

            # print(f"\nCluster {cid}")
            # print("Top keywords:", ", ".join(keywords[cid]))
            # print("Generated description:\n", description)
    except Exception as e:
        print("Exception:", e)
    finally:
        db_session.close()

if __name__ == "__main__":
    main()