def load_styles() -> str:
    """Return the full CSS block for CareerAlign, wrapped in a <style> tag."""
    return """
    <style>

    /* ── Base ── */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; }

    /* ── Page header ── */
    .page-header {
        margin-bottom: 2.5rem;
        padding-bottom: 1.5rem;
        border-bottom: 0.5px solid #e2e1d9;
    }
    .page-header h1 {
        font-family: 'DM Serif Display', Georgia, serif;
        font-size: 2rem;
        font-weight: 400;
        letter-spacing: -0.02em;
        margin: 0 0 0.4rem;
        color: #1a1a18;
    }
    .page-header p {
        font-size: 0.9rem;
        color: #6b6b66;
        margin: 0;
        line-height: 1.6;
    }

    /* ── AI insight sidebar ── */
    .insight-card {
        background: #f8f7f4;
        border-radius: 12px;
        padding: 1.5rem;
    }
    .insight-eyebrow {
        font-size: 0.65rem;
        font-weight: 500;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #9b9b96;
        margin-bottom: 1rem;
    }
    .confidence-ring {
        text-align: center;
        margin-bottom: 1.25rem;
    }
    .confidence-pct {
        font-family: 'DM Serif Display', Georgia, serif;
        font-size: 2.25rem;
        font-weight: 400;
        color: #1a1a18;
        line-height: 1;
        display: block;
    }
    .confidence-label {
        font-size: 0.7rem;
        color: #9b9b96;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .role-name {
        font-family: 'DM Serif Display', Georgia, serif;
        font-size: 1.15rem;
        font-weight: 400;
        color: #1a1a18;
        line-height: 1.35;
        padding-bottom: 0.85rem;
        border-bottom: 0.5px solid #e2e1d9;
        margin-bottom: 0.85rem;
    }
    .match-summary {
        font-size: 0.8rem;
        color: #6b6b66;
        line-height: 1.65;
        margin-bottom: 1.1rem;
    }
    .alt-label {
        font-size: 0.65rem;
        font-weight: 500;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #9b9b96;
        margin-bottom: 0.3rem;
    }
    .alt-role {
        font-size: 0.85rem;
        font-weight: 500;
        color: #0F6E56;
        margin-bottom: 0.3rem;
    }
    .alt-why {
        font-size: 0.75rem;
        color: #6b6b66;
        line-height: 1.55;
    }

    /* ── Results header ── */
    .results-header {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        margin-bottom: 1rem;
    }
    .results-title {
        font-family: 'DM Serif Display', Georgia, serif;
        font-size: 1.35rem;
        font-weight: 400;
        color: #1a1a18;
    }
    .results-count {
        font-size: 0.75rem;
        color: #9b9b96;
    }

    /* ── Job card ── */
    .job-card {
        background: #ffffff;
        border: 0.5px solid #e2e1d9;
        border-radius: 12px;
        padding: 1.1rem 1.35rem;
        margin-bottom: 0.75rem;
        transition: border-color 0.18s;
    }
    .job-card:hover  { border-color: #b4b2a9; }
    .job-card.match-open { border-color: #0F6E56; }

    .card-top {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 0.65rem;
    }
    .card-title {
        font-size: 0.9rem;
        font-weight: 500;
        color: #1a1a18;
        margin: 0 0 0.2rem;
    }
    .card-org {
        font-size: 0.75rem;
        color: #9b9b96;
    }
    .card-score-block { text-align: right; flex-shrink: 0; }
    .card-score-num {
        font-family: 'DM Serif Display', Georgia, serif;
        font-size: 1.5rem;
        font-weight: 400;
        color: #1a1a18;
        line-height: 1;
        display: block;
        margin-bottom: 0.3rem;
    }

    /* ── Score bar ── */
    .score-track {
        height: 3px;
        background: #e8e7e0;
        border-radius: 2px;
        margin-bottom: 0;
        overflow: hidden;
    }
    .score-fill-strong { height: 3px; background: #0F6E56; border-radius: 2px; }
    .score-fill-mod    { height: 3px; background: #BA7517; border-radius: 2px; }
    .score-fill-weak   { height: 3px; background: #D85A30; border-radius: 2px; }

    /* ── Badges ── */
    .badge {
        display: inline-block;
        font-size: 0.65rem;
        font-weight: 500;
        letter-spacing: 0.04em;
        padding: 3px 9px;
        border-radius: 20px;
    }
    .badge-strong { background: #E1F5EE; color: #085041; }
    .badge-mod    { background: #FAEEDA; color: #633806; }
    .badge-weak   { background: #FAECE7; color: #712B13; }

    /* ── Keyword chips ── */
    .kw-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.75rem;
        margin: 0.9rem 0;
    }
    .kw-head {
        font-size: 0.65rem;
        font-weight: 500;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #9b9b96;
        margin-bottom: 0.4rem;
    }
    .chips { display: flex; flex-wrap: wrap; gap: 4px; }
    .chip {
        font-size: 0.7rem;
        padding: 3px 9px;
        border-radius: 20px;
        line-height: 1.4;
    }
    .chip-have { background: #E1F5EE; color: #085041; }
    .chip-miss { background: #FAECE7; color: #712B13; }

    /* ── Card divider & description ── */
    .card-divider {
        height: 0.5px;
        background: #e8e7e0;
        margin: 0.75rem 0;
    }
    .job-desc {
        font-size: 0.78rem;
        color: #6b6b66;
        line-height: 1.7;
    }

    /* ── Custom test section ── */
    .test-section {
        margin-top: 2.5rem;
        padding-top: 1.5rem;
        border-top: 0.5px solid #e2e1d9;
    }
    .test-header {
        font-family: 'DM Serif Display', Georgia, serif;
        font-size: 1.15rem;
        font-weight: 400;
        margin-bottom: 0.4rem;
        color: #1a1a18;
    }
    .test-sub {
        font-size: 0.8rem;
        color: #6b6b66;
        margin-bottom: 1rem;
        line-height: 1.6;
    }

    /* ── Streamlit button override ── */
    .stButton > button {
        background: #0F6E56 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        padding: 0.55rem 1.4rem !important;
        transition: background 0.15s !important;
    }
    .stButton > button:hover { background: #0a5240 !important; }

    </style>
    """