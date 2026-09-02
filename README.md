🎯 AI Job Scraper & Resume Matcher

An MVP that scrapes job postings, parses PDF resumes, and uses AI to generate a structured match analysis (score, matching/missing skills, strengths, gaps) plus a tailored cover letter — via a Streamlit UI.

Built as a 1-week sprint MVP.

🔗 Live Demo: https://<your-app-name>.streamlit.app — pre-configured with API keys, no setup needed to try it.

Features
Upload a PDF resume → text extracted via pdfplumber.
Provide a job posting URL (scraped via requests + BeautifulSoup) or paste a JD manually.
Dual AI engine: Groq (llama-3.3-70b-versatile) as primary, Google Gemini (gemini-2.5-flash) as automatic fallback if Groq fails or rate-limits (429/5xx).
Structured, validated output (via pydantic): match score, matching/missing skills, strengths, gaps, summary.
Visual score gauge + color-coded skill pills.
AI-generated, downloadable cover letter (selectable tone).
How the fallback works

Every LLM call in matcher.py tries Groq first. If it fails, rate-limits, or returns bad JSON, the app automatically retries with Gemini — no crash, no manual intervention. The UI shows which engine actually answered (engine_used), so this is easy to verify live. The app only errors out if both engines fail.

Setup
bash
git clone https://github.com/<your-username>/ai-job-scraper-resume-matcher.git
cd ai-job-scraper-resume-matcher
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py

Opens at http://localhost:8501.

API Keys

The app auto-detects how it's being run:

On the hosted demo link — keys are pre-set via Streamlit Cloud Secrets. You'll see a "🔒 Running with pre-configured API keys" badge and can use it immediately.
Running locally — no secrets file is found, so the sidebar shows manual key inputs instead. Get free keys: Groq Console · Google AI Studio. Keys stay in session memory only, never written to disk.

To deploy your own copy: push to GitHub, deploy on share.streamlit.io, then add your keys under App settings → Secrets:

toml
GROQ_API_KEY = "gsk_..."
GEMINI_API_KEY = "AIza..."
Usage
Upload a PDF resume → Extract Resume Text.
Provide a job URL (Scrape Job Description) or paste a JD manually.
Click Run Match Analysis → review score, skills, strengths/gaps.
Optionally generate and download a tailored cover letter.
Known limitations
PDF resumes only (no DOCX/TXT yet).
JS-heavy job boards (e.g. LinkedIn) may not scrape cleanly — paste manually if needed.
No persistence (results reset on refresh) and no authentication on the hosted link.
If both Groq and Gemini fail or hit quota, the request fails with a combined error.

Next up (post-MVP): DOCX support, match history storage, batch matching, auth on the hosted deploy, a third fallback engine.

Tech stack

Streamlit + Plotly · pdfplumber · BeautifulSoup4 · Groq (llama-3.3-70b-versatile) · Gemini (gemini-2.5-flash) · pydantic

License

MIT
