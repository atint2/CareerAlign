import streamlit as st
import html

# ── Static HTML blocks ────────────────────────────────────────────────────────

def render_page_header() -> None:
    st.markdown("""
    <div class="page-header">
      <h1>CareerAlign</h1>
      <p>Upload your resume and our AI will surface the roles that fit you best —<br>
         ranked by match strength with keyword-level detail.</p>
    </div>
    """, unsafe_allow_html=True)

def render_test_section() -> None:
    st.markdown("""
    <div class="test-section">
      <div class="test-header">Test with a custom job description</div>
      <div class="test-sub">
        Paste any job posting below to see how your resume measures up.
        Include responsibilities, qualifications, and required skills for the best results.
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Utility builders ──────────────────────────────────────────────────────────

def _badge_props(pct: int, strong: int, moderate: int) -> tuple[str, str, str]:
    """Return (label, badge_class, fill_class) for a given score percentage."""
    if pct >= strong:
        return "Strong match", "badge-strong", "score-fill-strong"
    elif pct >= moderate:
        return "Moderate match", "badge-mod",    "score-fill-mod"
    else:
        return "Weak match",     "badge-weak",   "score-fill-weak"

def _chips_html(keywords: list[str], chip_class: str) -> str:
    if not keywords:
        return '<span style="font-size:0.75rem;color:#9b9b96;">—</span>'
    return "".join(
        f'<span class="chip {chip_class}">{kw}</span>'
        for kw in keywords
    )

def _confidence_arc(pct: int) -> str:
    """SVG semicircle showing confidence percentage."""
    circumference = 3.14159 * 34  # π × radius (half-circle arc length)
    offset = circumference * (1 - pct / 100)
    return f"""
    <div class="confidence-ring">
      <svg width="86" height="50" viewBox="0 0 86 50" style="overflow:visible">
        <path d="M8 46 A34 34 0 0 1 78 46"
              fill="none" stroke="#e2e1d9" stroke-width="5" stroke-linecap="round"/>
        <path d="M8 46 A34 34 0 0 1 78 46"
              fill="none" stroke="#0F6E56" stroke-width="5" stroke-linecap="round"
              stroke-dasharray="{circumference:.1f}"
              stroke-dashoffset="{offset:.1f}"/>
      </svg>
      <span class="confidence-pct">{pct}%</span>
      <span class="confidence-label">confidence</span>
    </div>
    """

# ── Main render functions ─────────────────────────────────────────────────────

def render_job_card(job: dict, thresholds: tuple[int, int] = (70, 40)) -> None:
    description = html.escape(job.get('description', ''))

    pct = int(job["similarity"] * 100)
    strong_t, mod_t = thresholds
    label, badge_cls, fill_cls = _badge_props(pct, strong_t, mod_t)

    have_chips = _chips_html(job.get("top_keywords", []), "chip-have")
    miss_chips = _chips_html(job.get("missing_keywords", []), "chip-miss")

    # Card open + header
    with st.expander(f"{job['title']} — {pct}%"):
        st.markdown(f"""
        <div class="card-score-block" style="margin-bottom:0.75rem">
          <span class="badge {badge_cls}">{label}</span>
        </div>
        <div class="score-track">
          <div class="{fill_cls}" style="width:{pct}%"></div>
        </div>
        <div class="kw-grid">
          <div>
            <div class="kw-head">You have</div>
            <div class="chips">{have_chips}</div>
          </div>
          <div>
            <div class="kw-head">You're missing</div>
            <div class="chips">{miss_chips}</div>
          </div>
        </div>
        <div class="card-divider"></div>
        <p class="job-desc">{description}</p>
        """, unsafe_allow_html=True)

def render_insight_sidebar(insights: dict) -> None:
    pct     = insights.get("confidence_score", 0)
    arc     = _confidence_arc(pct)
    alt_role = insights.get("alternative_role", "")
    alt_why  = insights.get("alternative_role_suggestions", "")

    alt_html = ""
    if alt_role:
        alt_html = f"""
        <div class="alt-label">Consider also</div>
        <div class="alt-role">{alt_role}</div>
        <div class="alt-why">{alt_why}</div>
        """

    st.markdown(f"""
    <div class="insight-card">
      <div class="insight-eyebrow">AI insight</div>
      {arc}
      <div class="role-name">{insights.get('recommended_job_title', '—')}</div>
      <p class="match-summary">{insights.get('match_summary', '')}</p>
      {alt_html}
    </div>
    """, unsafe_allow_html=True)

def render_match_section(
    title: str,
    matches: list[dict],
    thresholds: tuple[int, int] = (70, 40),
) -> None:
    count = len(matches)
    st.markdown(f"""
    <div class="results-header">
      <span class="results-title">{title}</span>
      <span class="results-count">{count} result{"s" if count != 1 else ""}</span>
    </div>
    """, unsafe_allow_html=True)

    for job in matches:
        render_job_card(job, thresholds)

def render_parsed_resume(resume: str) -> None:
    st.markdown("### Parsed Resume")
    st.text_area(
        label="Parsed Resume Text",
        value=resume,
        height=400,
        disabled=True
    )