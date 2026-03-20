import streamlit as st
import requests
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.services.file_reader import parse_with_llama
from ui.styles import load_styles
from ui.components import (
    render_page_header,
    render_insight_sidebar,
    render_match_section,
    render_test_section,
)

st.set_page_config(
    page_title="CareerAlign",
    page_icon=":briefcase:",
    layout="wide"
)

# Load font separately first
st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">',
    unsafe_allow_html=True
)

# Then load styles
st.markdown(load_styles(), unsafe_allow_html=True)

render_page_header()

uploaded_file = st.file_uploader(
    "Upload your resume",
    type=["pdf", "docx"],
    label_visibility="collapsed",
    help="PDF or DOCX, max 10 MB"
)

if uploaded_file:
    st.success(f"**{uploaded_file.name}** uploaded successfully.")

generate = st.button("Analyze my resume")

# ── Main results ──────────────────────────────────────────────────────────────

if generate:
    if not uploaded_file:
        st.error("Please upload your resume first.")
    else:
        with st.spinner("Analyzing your resume…"):
            resume_text = parse_with_llama(uploaded_file)
            response = requests.post(
                "http://localhost:8000/api/hybrid-match-resume/",
                json={"resume_text": resume_text}
            )

        if response.status_code != 200:
            st.error(f"Backend error {response.status_code}: {response.text}")
        else:
            data = response.json()
            insights = data.get("insights")

            sidebar_col, main_col = st.columns([1, 2.8], gap="large")

            with sidebar_col:
                if insights:
                    render_insight_sidebar(insights)
                else:
                    st.warning("AI insights unavailable.")

            with main_col:
                render_match_section(
                    "Career matches",
                    data.get("hybrid_matches", []),
                    thresholds=(50, 25),
                )

                postings = data.get("posting_matches", [])
                if postings:
                    st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)
                    render_match_section(
                        "Job postings",
                        postings,
                        thresholds=(70, 40),
                    )

# ── Custom job description tester ────────────────────────────────────────────

render_test_section()

custom_jd = st.text_area(
    "Job description",
    height=220,
    placeholder="Paste the job description here…",
    label_visibility="collapsed"
)

test_btn = st.button("Test this job description")

if test_btn:
    if not uploaded_file:
        st.error("Please upload your resume before testing.")
    elif not custom_jd.strip():
        st.error("Please enter a job description.")
    else:
        with st.spinner("Running matcher…"):
            resume_text = parse_with_llama(uploaded_file)
            response = requests.post(
                "http://localhost:8000/api/hybrid-match-resume/",
                json={"resume_text": resume_text, "job_desc": custom_jd}
            )

        if response.status_code != 200:
            st.error(f"Backend error {response.status_code}: {response.text}")
        else:
            data = response.json()
            col1, col2 = st.columns(2, gap="medium")

            with col1:
                render_match_section(
                    "Semantic match (SBERT)",
                    data.get("sbert_matches", []),
                    thresholds=(70, 40),
                )
            with col2:
                render_match_section(
                    "Keyword match (TF-IDF)",
                    data.get("tfidf_matches", []),
                    thresholds=(30, 10),
                )