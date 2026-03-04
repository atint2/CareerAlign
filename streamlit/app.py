import streamlit as st
import requests
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.services.file_reader import read_pdf, read_docx

# Setup the page configuration
st.set_page_config(
    page_title="CareerAlign",
    page_icon=":briefcase:",
    layout="wide")

st.title("CareerAlign: Your AI-Powered Career Path Finder")
st.write("Find your ideal career path with CareerAlign! Our AI-powered platform analyzes your resume to recommend the best career options for you. Start your journey towards a fulfilling career today!")

# Create a file uploader for the resume
uploaded_file = st.file_uploader("Upload your resume (PDF or DOCX)", type=["pdf", "docx"])
if uploaded_file is not None:
    st.success("Resume uploaded successfully!")

# Generate career recommendations
if st.button("Generate Career Recommendations"):
    if uploaded_file is not None:
        st.info("Analyzing your resume and generating recommendations...")

        # Read content of resume
        if uploaded_file.name.endswith(".pdf"):
            resume_text = read_pdf(uploaded_file)

        else:
            resume_text = read_docx(uploaded_file)

        # Generate results using matcher
        response = requests.post(
        "http://localhost:8000/api/match-resume/",
        json={"resume_text": resume_text}
        )

        # Recommendation results
        st.subheader("Recommended Career Paths:")

        if response.status_code == 200:
            data = response.json()

            st.subheader("TF-IDF Matches")
            for job in data["tfidf_matches"]:
                similarity_percent = int(job["similarity"] * 100)
                with st.expander(f"{job['title']} — {similarity_percent}% match"):
                    st.progress(similarity_percent)
                    st.write("### Job Description")
                    st.write(job["description"])
                    
            st.subheader("SBERT Matches")
            for job in data["sbert_matches"]:
                similarity_percent = int(job["similarity"] * 100)
                with st.expander(f"{job['title']} — {similarity_percent}% match"):
                    st.progress(similarity_percent)
                    st.write("### Job Description")
                    st.write(job["description"])

            st.subheader("AI Career Insights")
            insights = data.get("insights")

            if insights:
                st.subheader("Recommended Role")
                st.write(insights["recommended_job_title"])

                st.subheader("Confidence Score")
                confidence = insights["confidence_score"]
                st.progress(confidence)
                st.write(f"{confidence}% confidence")

                st.subheader("Why This Matches")
                st.write(insights["match_summary"])

                if insights.get("alternative_role"):
                    st.subheader("Better Fit Suggestion")
                    st.write(insights["alternative_role"])
            else:
                st.warning("AI insights unavailable.")

        else:
            st.error(f"Backend error: {response.status_code}")
            st.write(response.text)

    else:
        st.error("Please upload your resume to get recommendations.")