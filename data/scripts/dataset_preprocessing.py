import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

def clean_job_postings_dataset(dataset_filepath="data/raw/arshkon-linkedin-dataset.csv"):
    """
    Cleans the input dataset by removing entries with duplicate job descriptions and missing values for job title
    Also keeps only relevant columns.
    Cleaned dataset is saved to a new CSV file under data/processed.
    """
    df = pd.read_csv(dataset_filepath)

    # Remove duplicates based on 'desc_raw' column
    df_cleaned = df.drop_duplicates(subset=['description'])

    # Remove entries with missing job titles or descriptions
    df_cleaned = df_cleaned.dropna(subset=['title'])
    df_cleaned = df_cleaned.dropna(subset=['description'])

    # Keep only relevant columns
    relevant_columns = ['job_id', 'title', 'description', 'company_name', 'formatted_work_type', 'formatted_experience_level', 'skills_desc']
    df_cleaned = df_cleaned[relevant_columns]

    # Save cleaned dataset
    output_filepath = "data/processed/cleaned_job_postings.csv"
    df_cleaned.to_csv(output_filepath, index=False)
    print(f"Cleaned dataset saved to {output_filepath}")

def clean_resume_dataset(dataset_filepath="data/raw/resumes.csv"):
    """
    Cleans the input dataset by removing entries with duplicate resumes and missing values for Resume_str
    Also keeps only relevant columns.
    Cleaned dataset is saved to a new CSV file under data/processed.
    """
    df = pd.read_csv(dataset_filepath)

    # Remove duplicates based on 'Resume_str' column
    df_cleaned = df.drop_duplicates(subset=['Resume_str'])

    # Remove entries with missing resume text
    df_cleaned = df_cleaned.dropna(subset=['Resume_str'])

    # Keep only relevant columns
    relevant_columns = ['ID', 'Resume_str']
    df_cleaned = df_cleaned[relevant_columns]

    # Save cleaned dataset
    output_filepath = "data/processed/cleaned_resumes.csv"
    df_cleaned.to_csv(output_filepath, index=False)
    print(f"Cleaned dataset saved to {output_filepath}")

def clean_skills_dataset(dataset_filepath="data/raw/skills.csv"):
    """
    Cleans the input dataset by removing entries with duplicate skills and missing values for 'Example'
    Also keeps only relevant columns.
    Cleaned dataset is saved to a new CSV file under data/processed.
    """    
    df = pd.read_csv(dataset_filepath)

    # Remove duplicates based on 'Resume_str' column
    df_cleaned = df.drop_duplicates(subset=['Example'])

    # Remove entries with missing resume text
    df_cleaned = df_cleaned.dropna(subset=['Example'])

    # Keep only relevant columns
    relevant_columns = ['Example', 'Hot Technology', 'In Demand']
    df_cleaned = df_cleaned[relevant_columns]

    # Save cleaned dataset
    output_filepath = "data/processed/cleaned_skills.csv"
    df_cleaned.to_csv(output_filepath, index=False)
    print(f"Cleaned dataset saved to {output_filepath}")

# Run cleaning script
if __name__ == "__main__":
    clean_job_postings_dataset()
    clean_resume_dataset()
    clean_skills_dataset()
    

