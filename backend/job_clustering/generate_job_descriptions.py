import re
from google import genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from backend import database, models
from data.scripts.preprocessor_sbert import SBERTPreprocessor
from data.scripts.preprocessor_tfidf import TFIDFPreprocessor
from collections import defaultdict
import numpy as np 
import time

# Load API key
from dotenv import load_dotenv
import os
load_dotenv()
API_KEYS = os.getenv("GEMINI_API_KEYS").split(",")
API_KEY = API_KEYS[0]

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
    - Respond in the following format:
    **Role Title:** [Inferred common role title]
    **Professional Summary:** [Concise summary of common responsibilities and skills]

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
    Rotates through API keys on 429 RESOURCE_EXHAUSTED errors.
    """

    if not sample_titles and not sample_descriptions:
        return "Insufficient data to generate description."

    prompt = create_llm_prompt(keywords, sample_titles, sample_descriptions)

    for i, key in enumerate(API_KEYS):
        try:
            current_client = genai.Client(api_key=key)

            response = current_client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=prompt,
            )
            return response.text.strip()
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                print(f"API key {i+1}/{len(API_KEYS)} exhausted, trying next key...")
                if i == len(API_KEYS) - 1:
                    return "All API keys exhausted."
            else:
                print("LLM generation failed:", e)
                return "Description unavailable."

def main():
    # Create new database session instance
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
        
        # Initialize preprocessors
        tfidf_prep = TFIDFPreprocessor()
        sbert_prep = SBERTPreprocessor()

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
                db_session.query(models.ClusterExperimental)
                .filter(models.ClusterExperimental.cluster_id == int(cid))
                .one_or_none()
            )
            if existing:
                # Check if general job description already exists
                if existing.general_job_desc_raw:
                    # If it already exists, make sure it has been preprocessed for models
                    if not existing.general_job_desc_tfidf or not existing.general_job_desc_sbert:
                        existing.general_job_desc_tfidf = tfidf_prep.clean_text_tfidf(existing.general_job_desc_raw)
                        existing.general_job_desc_sbert = sbert_prep.clean_text_sbert(existing.general_job_desc_raw)
                        db_session.commit()
                    else:
                        pass
                else:
                    # Generate description for cluster
                    description = generate_job_description(
                        keywords[cid],
                        sample_titles,
                        sample_descs,
                    )

                    # Extract information from generated description using regex
                    # Extract title from cluster_desc using regex
                    title = None 
                    title_match = re.search(r"\*\*Role Title:\*\*\s*(.+)", description) 
                    if title_match: 
                        title = title_match.group(1).strip() 
                    # Extract general job description from cluster_desc using regex 
                    desc_match = re.search(r"\*\*Professional Summary:\*\*\s*(.+)", description, re.DOTALL) 
                    if desc_match: 
                        description = desc_match.group(1).strip() 
                        # Preprocess description for TF-IDF and SBERT
                        tfidf_description = tfidf_prep.clean_text_tfidf(description)
                        sbert_description = sbert_prep.clean_text_sbert(description)
                    else: 
                        print(f"Warning: Could not extract description for cluster_id {cid}") 

                    if description == "All API keys exhausted.":
                        print("All API keys exhausted. Breaking...")
                        break
                    
                    # Save description to database
                    existing.general_job_desc_raw = description
                    existing.general_job_desc_tfidf = tfidf_description
                    existing.general_job_desc_sbert = sbert_description
                    existing.title = title
                    db_session.commit()
                    print(f"Description for cid {cid} successfully saved to the database")
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