import streamlit as st
import requests
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.services.file_reader import parse_with_llama

# Setup the page configuration
st.set_page_config(
    page_title="CareerAlign",
    page_icon=":briefcase:",
    layout="wide")

# Custom CSS for progress bars (how strong/weak match is)
st.markdown("""
    <style>
    /* Strong - Green */
    .strong-match div[data-testid="stProgressBar"] > div > div {
        background-color: #16a34a;
    }

    /* Moderate - Yellow */
    .moderate-match div[data-testid="stProgressBar"] > div > div {
        background-color: #eab308;
    }

    /* Weak - Orange */
    .weak-match div[data-testid="stProgressBar"] > div > div {
        background-color: #f97316;
    }
    </style>
    """, unsafe_allow_html=True
)

def get_match_badge(similarity_percent, strong_threshold=70, moderate_threshold=40):
    """
    Returns a dict with match classification info: label, color, css_class.
    """
    if similarity_percent >= strong_threshold:
        return {"label": "Strong", "color": "#16a34a", "css_class": "strong-match"}
    elif similarity_percent >= moderate_threshold:
        return {"label": "Moderate", "color": "#eab308", "css_class": "moderate-match"}
    else:
        return {"label": "Weak", "color": "#f97316", "css_class": "weak-match"}

def render_match_section(title, matches, strong_threshold=70, moderate_threshold=40):
    """
    Helper function for rendering match sections with progress bars
    """
    st.subheader(title)

    for job in matches:
        similarity_percent = int(job["similarity"] * 100)
        badge = get_match_badge(similarity_percent, strong_threshold, moderate_threshold)

        with st.expander(
            f"{job['title']} — {similarity_percent}% ({badge['label']})"
        ):
            # Badge
            st.markdown(f"""
                <div style="
                    display:inline-block;
                    padding:6px 12px;
                    border-radius:8px;
                    background-color:{badge['color']};
                    color:white;
                    font-weight:600;">
                    {badge['label']} Match
                </div>
                """, 
                unsafe_allow_html=True
            )

            # Colored progress bar wrapper
            st.markdown(f'<div class="{badge['css_class']}">', unsafe_allow_html=True)
            st.progress(similarity_percent / 100)
            st.markdown("</div>", unsafe_allow_html=True)

            st.write("### Job Description")
            st.write(job["description"])

            st.write("### Top Keywords in Job Description Relevant to Your Resume")
            if job.get("top_keywords"):
                st.write(", ".join(job["top_keywords"]))

# Main app interface
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
        resume_text = parse_with_llama(uploaded_file)

        # Generate results using matcher
        response = requests.post(
        "http://localhost:8000/api/match-resume/",
        json={"resume_text": resume_text}
        )

        # Recommendation results
        st.subheader("Recommended Career Paths:")

        if response.status_code == 200:
            data = response.json()

            render_match_section(
                "SBERT Matches",
                data["sbert_matches"],
                strong_threshold=70,
                moderate_threshold=40
            )

            render_match_section(
                "TF-IDF Matches",
                data["tfidf_matches"],
                strong_threshold=30,      
                moderate_threshold=10
            )

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

                if insights.get("alternative_role_suggestions"):
                    st.subheader("Why This Might be Better")
                    st.write(insights["alternative_role_suggestions"])

            else:
                st.warning("AI insights unavailable.")
        else:
            st.error(f"Backend error: {response.status_code}")
            st.write(response.text)
    else:
        st.error("Please upload your resume to get recommendations.")

# Allow users to enter a job description directly for testing
st.write("---")
st.subheader("Test with Custom Job Description")
# Create a file uploader for the resume
custom_job_desc = st.text_area("Enter a job description to test the matcher:", height=300)
if st.button("Test Custom Job Description"):
    if uploaded_file is not None and custom_job_desc.strip():
        st.info("Analyzing your resume and generating recommendations...")

        # Read content of resume
        if uploaded_file.name.endswith(".pdf"):
            resume_text = read_pdf(uploaded_file)

        else:
            resume_text = read_docx(uploaded_file)
    
        st.info("Testing matcher with custom job description...")

        # Generate results using matcher
        response = requests.post(
        "http://localhost:8000/api/match-resume/",
        json={"resume_text": resume_text, "job_desc": custom_job_desc}
        )

        if response.status_code == 200:
            data = response.json()
            sbert_similarity = int(data['sbert_matches'][0]['similarity'] * 100)
            tfidf_similarity = int(data['tfidf_matches'][0]['similarity'] * 100)

            st.write("Similarity Score with Custom Job Description:")
            
            # SBERT
            sbert_badge = get_match_badge(sbert_similarity, 70, 40)

            st.title(f"{sbert_similarity}% (SBERT)")
            # Badge
            st.markdown(f"""
                <div style="
                    display:inline-block;
                    padding:6px 12px;
                    border-radius:8px;
                    background-color:{sbert_badge['color']};
                    color:white;
                    font-weight:600;">
                    {sbert_badge['label']} Match
                </div>
                """, 
                unsafe_allow_html=True
            )
            # Colored progress bar wrapper
            st.markdown(f'<div class="{sbert_badge['css_class']}">', unsafe_allow_html=True)
            st.progress(sbert_similarity / 100)
            st.markdown("</div>", unsafe_allow_html=True)

            sbert_top_keywords = data['sbert_matches'][0]['top_keywords']
            if sbert_top_keywords:
                st.write("### Top Keywords in Job Description Relevant to Your Resume")
                st.write(", ".join(sbert_top_keywords))

            # TF-IDF
            tfidf_badge = get_match_badge(tfidf_similarity, 30, 10)
            
            st.title(f"{tfidf_similarity}% (TF-IDF)")
            # Badge
            st.markdown(f"""
                <div style="
                    display:inline-block;
                    padding:6px 12px;
                    border-radius:8px;
                    background-color:{tfidf_badge['color']};
                    color:white;
                    font-weight:600;">
                    {tfidf_badge['label']} Match
                </div>
                """, 
                unsafe_allow_html=True
            )
            # Colored progress bar wrapper
            st.markdown(f'<div class="{tfidf_badge['css_class']}">', unsafe_allow_html=True)
            st.progress(tfidf_similarity / 100)
            st.markdown("</div>", unsafe_allow_html=True)

            tfidf_top_keywords = data['tfidf_matches'][0]['top_keywords']
            if tfidf_top_keywords:
                st.write("### Top Keywords in Job Description Relevant to Your Resume")
                st.write(", ".join(tfidf_top_keywords))

        else:
            st.error(f"Backend error: {response.status_code}")
            st.write(response.text)
    else:
        st.error("Please enter a job description to test.")