import streamlit as st
import requests
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.services.file_reader import parse_with_llama
from ui.styles import load_styles
from ui.components import (
    render_page_header,
    render_insight_sidebar,
    render_match_section,
    render_test_section,
    render_parsed_resume
)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CareerAlign",
    page_icon=":briefcase:",
    layout="wide"
)

# ── Load styles ───────────────────────────────────────────────────────────────

st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">',
    unsafe_allow_html=True
)
st.markdown(load_styles(), unsafe_allow_html=True)

render_page_header()

# ── Initialize session state ──────────────────────────────────────────────────

if "resume_text" not in st.session_state:
    st.session_state.resume_text = None

if "uploaded_file_id" not in st.session_state:       
    st.session_state.uploaded_file_id = None

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if "hybrid_data" not in st.session_state:
    st.session_state.hybrid_data = None

if "downstream_done" not in st.session_state:
    st.session_state.downstream_done = False

if "posting_data" not in st.session_state:
    st.session_state.posting_data = None

if "show_resume" not in st.session_state:
    st.session_state.show_resume = False

# ── File upload ───────────────────────────────────────────────────────────────

uploaded_file = st.file_uploader(
    "Upload your resume",
    type=["pdf", "docx"],
    label_visibility="collapsed",
    help="PDF or DOCX, max 10 MB"
)

# Parse resume ONLY once per upload
if uploaded_file:
    file_id = (uploaded_file.name, uploaded_file.size)

    if file_id != st.session_state.uploaded_file_id:

        with st.spinner("Parsing resume..."):
            st.session_state.resume_text = parse_with_llama(uploaded_file)
            st.session_state.uploaded_file_id = file_id # Update the uploaded file ID to prevent re-parsing on reruns

        # Reset downstream state on new upload
        st.session_state.analysis_done = False
        st.session_state.downstream_done = False
        st.session_state.hybrid_data = None
        st.session_state.posting_data = None

# ── View my parsed resume button ────────────────────────────────────────────────────────────

if st.button("View my parsed resume"):
    st.session_state.show_resume = not st.session_state.show_resume
    if not uploaded_file:
        st.error("Please upload your resume first.")

if st.session_state.show_resume and st.session_state.resume_text:
    render_parsed_resume(st.session_state.resume_text)

# ── Analyze button ────────────────────────────────────────────────────────────

if st.button("Analyze my resume"):
    if not uploaded_file:
        st.error("Please upload your resume first.")
    else:
        with st.spinner("Analyzing your resume…"):
            try:
                response = requests.post(
                    "http://localhost:8000/api/hybrid-match-resume/",
                    json={"resume_text": st.session_state.resume_text},
                    timeout=120
                )
                response.raise_for_status()
                st.session_state.hybrid_data = response.json()
                st.session_state.analysis_done = True

            except requests.exceptions.RequestException as e:
                st.error(f"Connection error: {e}")

# ── Render main results ───────────────────────────────────────────────────────

if st.session_state.analysis_done and st.session_state.hybrid_data:

    data = st.session_state.hybrid_data
    insights = data.get("insights")
    hybrid_matches = data.get("hybrid_matches", [])

    sidebar_col, main_col = st.columns([1, 2.8], gap="large")

    with sidebar_col:
        if insights:
            render_insight_sidebar(insights)
        else:
            st.warning("AI insights unavailable.")

    with main_col:
        render_match_section(
            "Career matches",
            hybrid_matches,
            thresholds=(50, 25),
        )

        # ── Downstream button ───────────────────────────────────────────────

        if st.button(
            "Continue analysis with job postings",
            disabled=st.session_state.downstream_done
        ):
            with st.spinner("Analyzing job postings…"):
                try:
                    response = requests.post(
                        "http://localhost:8000/api/downstream-match-resume/",
                        json={
                            "resume_text": st.session_state.resume_text,
                            "hybrid_matches": hybrid_matches
                        },
                        timeout=480
                    )
                    response.raise_for_status()
                    st.session_state.posting_data = response.json()
                    st.session_state.downstream_done = True

                except requests.exceptions.RequestException as e:
                    st.error(f"Connection error: {e}")

        # ── Show downstream results if available ────────────────────────────

    if st.session_state.downstream_done and st.session_state.posting_data:
        data = st.session_state.posting_data
        insights = data.get("insights")
        posting_matches = data.get("posting_matches", [])

        sidebar_col, main_col = st.columns([1, 2.8], gap="large")

        with sidebar_col:
            if insights:
                render_insight_sidebar(insights)
            else:
                st.warning("AI insights unavailable.")

        with main_col:
            if posting_matches:
                st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)
                render_match_section(
                    "Job postings",
                    posting_matches,
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

if st.button("Test this job description"):
    if not uploaded_file:
        st.error("Please upload your resume before testing.")
    elif not custom_jd.strip():
        st.error("Please enter a job description.")
    else:
        with st.spinner("Running matcher…"):
            try:
                response = requests.post(
                    "http://localhost:8000/api/hybrid-match-resume/",
                    json={
                        "resume_text": st.session_state.resume_text,
                        "job_desc": custom_jd
                    },
                    timeout=120
                )
                response.raise_for_status()
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

            except requests.exceptions.RequestException as e:
                st.error(f"Connection error: {e}")