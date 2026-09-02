"""
app.py
-------
Streamlit UI for the AI Job Scraper & Resume Matcher MVP.

Features:
- Sidebar inputs for Groq (primary) and Gemini (fallback) API keys.
- Resume upload (PDF) and job input via URL scrape or manual paste.
- Runs dual-engine matching (matcher.py) with automatic Groq -> Gemini fallback.
- Displays match score as a gauge, matching/missing skills as pills, strengths/gaps, and summary.
- Cover letter generator using the same dual-engine fallback logic.
- Custom CSS for card-style sections, skill pills, and a hero header.
"""

import streamlit as st
import plotly.graph_objects as go

from scraper import extract_resume_text, scrape_job_description, ScraperError
from matcher import match_resume_to_job, generate_cover_letter, MatcherError, MatchResult

# --------------------------------------------------------------------------
# Page Config
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Job Scraper & Resume Matcher",
    page_icon="🎯",
    layout="wide",
)

# --------------------------------------------------------------------------
# Custom CSS — Hero header, cards, skill pills
# --------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
    /* Hero header */
    .hero-container {
        background: linear-gradient(135deg, #2E86AB 0%, #1B4965 100%);
        padding: 2.2rem 2rem;
        border-radius: 14px;
        margin-bottom: 1.8rem;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.25);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 0.3rem;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #DCEEF7;
        opacity: 0.9;
    }

    /* Card containers */
    .card {
        background-color: #1A1D27;
        border: 1px solid #2A2E3D;
        border-radius: 12px;
        padding: 1.4rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.15);
    }
    .card-title {
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
        color: #F5F6FA;
    }

    /* Skill pills */
    .pill {
        display: inline-block;
        padding: 0.35rem 0.9rem;
        margin: 0.25rem 0.35rem 0.25rem 0;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .pill-match {
        background-color: rgba(46, 204, 113, 0.15);
        color: #4ADE80;
        border: 1px solid rgba(74, 222, 128, 0.4);
    }
    .pill-missing {
        background-color: rgba(231, 76, 60, 0.15);
        color: #F87171;
        border: 1px solid rgba(248, 113, 113, 0.4);
    }
    .pill-empty {
        color: #8A8F9C;
        font-size: 0.85rem;
        font-style: italic;
    }

    /* Engine badge */
    .engine-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 8px;
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .engine-groq {
        background-color: rgba(46, 134, 171, 0.15);
        color: #5DB8E8;
        border: 1px solid rgba(93, 184, 232, 0.4);
    }
    .engine-gemini {
        background-color: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        border: 1px solid rgba(251, 191, 36, 0.4);
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_pills(skills: list, pill_class: str, empty_label: str) -> str:
    """Builds an HTML string of pill badges for a skill list, or an empty-state message."""
    if not skills:
        return f'<span class="pill-empty">{empty_label}</span>'
    pills_html = "".join(f'<span class="pill {pill_class}">{skill}</span>' for skill in skills)
    return pills_html


# --------------------------------------------------------------------------
# Session State Initialization
# --------------------------------------------------------------------------

if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "job_description" not in st.session_state:
    st.session_state.job_description = ""
if "match_result" not in st.session_state:
    st.session_state.match_result = None
if "cover_letter" not in st.session_state:
    st.session_state.cover_letter = ""
if "engine_used_for_match" not in st.session_state:
    st.session_state.engine_used_for_match = None


# --------------------------------------------------------------------------
# Sidebar — API Keys
# --------------------------------------------------------------------------

with st.sidebar:
    st.header("🔑 API Configuration")

    # Check for host-configured secrets first (e.g., Streamlit Community Cloud
    # secrets.toml). These are never committed to GitHub — set via the deploy
    # platform's dashboard. If present, the app runs "pre-configured" and hides
    # the input fields. Otherwise, it falls back to manual entry (BYOK mode).
    secret_groq_key = st.secrets.get("GROQ_API_KEY", "")
    secret_gemini_key = st.secrets.get("GEMINI_API_KEY", "")

    using_hosted_keys = bool(secret_groq_key or secret_gemini_key)

    if using_hosted_keys:
        st.success("🔒 Running with pre-configured API keys. No setup needed.")
        groq_api_key = secret_groq_key
        gemini_api_key = secret_gemini_key

        with st.expander("Use your own keys instead"):
            override_groq = st.text_input(
                "Groq API Key (Primary)", type="password", placeholder="gsk_...",
                key="override_groq",
            )
            override_gemini = st.text_input(
                "Gemini API Key (Fallback)", type="password", placeholder="AIza...",
                key="override_gemini",
            )
            if override_groq:
                groq_api_key = override_groq
            if override_gemini:
                gemini_api_key = override_gemini

    else:
        st.caption("Keys are used only for this session and never stored.")

        groq_api_key = st.text_input(
            "Groq API Key (Primary)",
            type="password",
            placeholder="gsk_...",
            help="Powers llama-3.3-70b-versatile as the primary matching engine.",
        )

        gemini_api_key = st.text_input(
            "Gemini API Key (Fallback)",
            type="password",
            placeholder="AIza...",
            help="Powers gemini-2.5-flash. Used automatically if Groq fails or rate-limits.",
        )

        st.divider()

        if groq_api_key and gemini_api_key:
            st.success("Both engines configured. Full fallback protection active.")
        elif groq_api_key and not gemini_api_key:
            st.warning("Only Groq configured. No fallback if Groq fails.")
        elif not groq_api_key and gemini_api_key:
            st.warning("Only Gemini configured. Will be used directly (no primary).")
        else:
            st.error("No API keys configured. Enter at least one to proceed.")

        st.divider()
        st.caption("Get keys: [Groq Console](https://console.groq.com/keys) · "
                   "[Google AI Studio](https://aistudio.google.com/apikey)")


# --------------------------------------------------------------------------
# Hero Header
# --------------------------------------------------------------------------

st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">🎯 AI Job Scraper & Resume Matcher</div>
        <div class="hero-subtitle">Upload your resume, provide a job posting, and get an instant AI-powered match analysis.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

col_left, col_right = st.columns(2)

# --------------------------------------------------------------------------
# Left Column — Resume Upload
# --------------------------------------------------------------------------

with col_left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📄 Your Resume</div>', unsafe_allow_html=True)

    uploaded_resume = st.file_uploader(
        "Upload PDF resume",
        type=["pdf"],
        help="Only PDF format is supported in this MVP.",
        label_visibility="collapsed",
    )

    if uploaded_resume is not None:
        if st.button("Extract Resume Text", use_container_width=True):
            with st.spinner("Extracting text from PDF..."):
                try:
                    st.session_state.resume_text = extract_resume_text(uploaded_resume)
                    st.success(f"Extracted {len(st.session_state.resume_text)} characters.")
                except ScraperError as e:
                    st.error(str(e))

    if st.session_state.resume_text:
        with st.expander("View extracted resume text", expanded=False):
            st.text_area(
                "Resume text",
                value=st.session_state.resume_text,
                height=250,
                label_visibility="collapsed",
            )

    st.markdown('</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Right Column — Job Description
# --------------------------------------------------------------------------

with col_right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">💼 Job Description</div>', unsafe_allow_html=True)

    job_input_mode = st.radio(
        "Input method",
        options=["Scrape from URL", "Paste manually"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if job_input_mode == "Scrape from URL":
        job_url = st.text_input("Job posting URL", placeholder="https://company.com/careers/job-123")
        if st.button("Scrape Job Description", use_container_width=True):
            with st.spinner("Fetching and parsing job posting..."):
                try:
                    st.session_state.job_description = scrape_job_description(job_url)
                    st.success(f"Scraped {len(st.session_state.job_description)} characters.")
                except ScraperError as e:
                    st.error(str(e))
                    st.info("Tip: You can switch to 'Paste manually' if scraping fails.")
    else:
        pasted_jd = st.text_area(
            "Paste job description",
            height=200,
            placeholder="Paste the full job description text here...",
        )
        if st.button("Save Job Description", use_container_width=True):
            if pasted_jd.strip():
                st.session_state.job_description = pasted_jd.strip()
                st.success(f"Saved {len(st.session_state.job_description)} characters.")
            else:
                st.error("Please paste some text first.")

    if st.session_state.job_description:
        with st.expander("View job description text", expanded=False):
            st.text_area(
                "Job description text",
                value=st.session_state.job_description,
                height=250,
                label_visibility="collapsed",
            )

    st.markdown('</div>', unsafe_allow_html=True)


st.divider()

# --------------------------------------------------------------------------
# Run Match Analysis
# --------------------------------------------------------------------------

run_disabled = not (
    st.session_state.resume_text
    and st.session_state.job_description
    and (groq_api_key or gemini_api_key)
)

if st.button("🚀 Run Match Analysis", type="primary", use_container_width=True, disabled=run_disabled):
    with st.spinner("Analyzing match... (Groq primary, Gemini fallback if needed)"):
        try:
            result: MatchResult = match_resume_to_job(
                resume_text=st.session_state.resume_text,
                job_description=st.session_state.job_description,
                groq_api_key=groq_api_key,
                gemini_api_key=gemini_api_key,
            )
            st.session_state.match_result = result
            st.session_state.engine_used_for_match = result.engine_used
        except MatcherError as e:
            st.session_state.match_result = None
            st.error(f"Matching failed: {e}")

if run_disabled:
    st.caption("⬆️ Upload a resume, provide a job description, and enter at least one API key to enable analysis.")


# --------------------------------------------------------------------------
# Results Display
# --------------------------------------------------------------------------

if st.session_state.match_result:
    result = st.session_state.match_result

    engine_class = "engine-groq" if result.engine_used == "groq" else "engine-gemini"
    engine_label = "🟢 Groq (Primary)" if result.engine_used == "groq" else "🟡 Gemini (Fallback)"
    st.markdown(
        f'<div class="engine-badge {engine_class}">Analysis completed using: {engine_label}</div>',
        unsafe_allow_html=True,
    )

    gauge_col, details_col = st.columns([1, 1.5])

    with gauge_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result.match_score,
            title={"text": "Match Score", "font": {"size": 22, "color": "#F5F6FA"}},
            number={"suffix": "%", "font": {"size": 40, "color": "#F5F6FA"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#8A8F9C"},
                "bar": {"color": "#2E86AB"},
                "bgcolor": "#1A1D27",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "#4A2A2A"},
                    {"range": [40, 70], "color": "#4A3F1F"},
                    {"range": [70, 100], "color": "#1F4A2E"},
                ],
                "threshold": {
                    "line": {"color": "#FFFFFF", "width": 3},
                    "thickness": 0.8,
                    "value": result.match_score,
                },
            },
        ))
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#F5F6FA"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with details_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Summary</div>', unsafe_allow_html=True)
        st.write(result.summary)

        st.markdown('<div class="card-title" style="margin-top:1rem;">✅ Matching Skills</div>', unsafe_allow_html=True)
        st.markdown(render_pills(result.matching_skills, "pill-match", "None identified."), unsafe_allow_html=True)

        st.markdown('<div class="card-title" style="margin-top:1rem;">❌ Missing Skills</div>', unsafe_allow_html=True)
        st.markdown(render_pills(result.missing_skills, "pill-missing", "None identified."), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    strength_col, gap_col = st.columns(2)
    with strength_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">💪 Strengths</div>', unsafe_allow_html=True)
        if result.strengths:
            for s in result.strengths:
                st.markdown(f"- {s}")
        else:
            st.markdown('<span class="pill-empty">None identified.</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with gap_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">⚠️ Gaps</div>', unsafe_allow_html=True)
        if result.gaps:
            for g in result.gaps:
                st.markdown(f"- {g}")
        else:
            st.markdown('<span class="pill-empty">None identified.</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # ----------------------------------------------------------------------
    # Cover Letter Generator
    # ----------------------------------------------------------------------

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">✍️ Cover Letter Generator</div>', unsafe_allow_html=True)

    tone = st.selectbox(
        "Tone",
        options=["professional", "enthusiastic", "concise", "formal"],
        index=0,
    )

    if st.button("Generate Cover Letter", use_container_width=True):
        with st.spinner("Writing your cover letter... (Groq primary, Gemini fallback if needed)"):
            try:
                letter = generate_cover_letter(
                    resume_text=st.session_state.resume_text,
                    job_description=st.session_state.job_description,
                    groq_api_key=groq_api_key,
                    gemini_api_key=gemini_api_key,
                    tone=tone,
                )
                st.session_state.cover_letter = letter
            except MatcherError as e:
                st.error(f"Cover letter generation failed: {e}")

    if st.session_state.cover_letter:
        st.text_area(
            "Generated Cover Letter",
            value=st.session_state.cover_letter,
            height=300,
        )
        st.download_button(
            label="📥 Download Cover Letter (.txt)",
            data=st.session_state.cover_letter,
            file_name="cover_letter.txt",
            mime="text/plain",
            use_container_width=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.caption("Run a match analysis above to see your score, skills breakdown, and cover letter generator.")
