"""
matcher.py
-----------
Core AI matching engine for the Job Scraper & Resume Matcher.

Architecture:
- Primary Engine: Groq API (llama-3.3-70b-versatile) — fast, low-latency structured JSON output.
- Fallback Engine: Google Gemini API (gemini-2.5-flash) — automatically invoked if Groq fails,
  rate-limits (429), or errors server-side (5xx).

The public entry point `match_resume_to_job()` always returns a validated MatchResult object,
regardless of which underlying engine served the request.
"""

import json
import logging
from typing import Optional, List

from pydantic import BaseModel, Field, ValidationError

from groq import Groq
from groq import APIStatusError as GroqAPIStatusError
from groq import APIConnectionError as GroqAPIConnectionError
from groq import RateLimitError as GroqRateLimitError

from google import genai
from google.genai import types as genai_types
from google.genai.errors import APIError as GeminiAPIError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GROQ_MODEL = "llama-3.3-70b-versatile"
GEMINI_MODEL = "gemini-2.5-flash"

# Groq status codes that should trigger a fallback to Gemini
GROQ_FALLBACK_STATUS_CODES = {429, 500, 502, 503, 504}


# --------------------------------------------------------------------------
# Structured Output Schema
# --------------------------------------------------------------------------

class MatchResult(BaseModel):
    """Validated structure for the resume-to-job match analysis."""
    match_score: int = Field(..., ge=0, le=100, description="Overall match score from 0-100")
    matching_skills: List[str] = Field(default_factory=list, description="Skills present in both resume and job")
    missing_skills: List[str] = Field(default_factory=list, description="Required skills missing from resume")
    strengths: List[str] = Field(default_factory=list, description="Key strengths of the candidate for this role")
    gaps: List[str] = Field(default_factory=list, description="Key gaps or weaknesses for this role")
    summary: str = Field(..., description="2-3 sentence summary of the overall fit")
    engine_used: str = Field(default="unknown", description="Which AI engine produced this result")


class MatcherError(Exception):
    """Raised when both Groq and Gemini fail to produce a usable result."""
    pass


# --------------------------------------------------------------------------
# Prompt Construction
# --------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert technical recruiter and resume analyst. "
    "You compare a candidate's resume against a job description and produce a "
    "strict, honest, structured match analysis. Respond ONLY with valid JSON. "
    "Do not include markdown code fences, explanations, or any text outside the JSON object."
)

JSON_SCHEMA_INSTRUCTIONS = """
Return a JSON object with EXACTLY this structure:
{
  "match_score": <integer 0-100>,
  "matching_skills": [<list of strings - skills found in both resume and job description>],
  "missing_skills": [<list of strings - required skills in job description but absent from resume>],
  "strengths": [<list of strings - candidate's key strengths relevant to this role>],
  "gaps": [<list of strings - candidate's key gaps or weaknesses relevant to this role>],
  "summary": "<2-3 sentence honest summary of overall fit>"
}
Rules:
- match_score must be a realistic integer reflecting true alignment, not inflated.
- Base every field strictly on the provided resume and job description text.
- Do not invent skills or experience not present in the resume.
- Output ONLY the raw JSON object, nothing else.
"""


def _build_user_prompt(resume_text: str, job_description: str) -> str:
    return (
        f"{JSON_SCHEMA_INSTRUCTIONS}\n\n"
        f"--- RESUME ---\n{resume_text.strip()}\n\n"
        f"--- JOB DESCRIPTION ---\n{job_description.strip()}\n"
    )


def _extract_json_object(raw_text: str) -> dict:
    """
    Extracts a JSON object from raw LLM output, tolerating stray markdown
    fences or leading/trailing whitespace/text that some models add despite instructions.
    """
    text = raw_text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    # Fallback: locate the first '{' and last '}' to isolate the JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in model output.")

    json_str = text[start:end + 1]
    return json.loads(json_str)


# --------------------------------------------------------------------------
# Groq Engine
# --------------------------------------------------------------------------

def _call_groq(api_key: str, resume_text: str, job_description: str) -> dict:
    """
    Calls the Groq API (llama-3.3-70b-versatile) and returns a parsed JSON dict.

    Raises:
        GroqRateLimitError, GroqAPIStatusError, GroqAPIConnectionError: on API failure
        ValueError / json.JSONDecodeError: on malformed output
    """
    client = Groq(api_key=api_key)

    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(resume_text, job_description)},
        ],
        temperature=0.2,
        max_tokens=2048,
        response_format={"type": "json_object"},
    )

    raw_content = completion.choices[0].message.content
    parsed = _extract_json_object(raw_content)
    return parsed


# --------------------------------------------------------------------------
# Gemini Engine (Fallback)
# --------------------------------------------------------------------------

def _call_gemini(api_key: str, resume_text: str, job_description: str) -> dict:
    """
    Calls the Google Gemini API (gemini-2.5-flash) and returns a parsed JSON dict.

    Raises:
        GeminiAPIError: on API failure
        ValueError / json.JSONDecodeError: on malformed output
    """
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=_build_user_prompt(resume_text, job_description),
        config=genai_types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,
            max_output_tokens=2048,
            response_mime_type="application/json",
        ),
    )

    raw_content = response.text
    if not raw_content:
        raise ValueError("Gemini returned an empty response.")

    parsed = _extract_json_object(raw_content)
    return parsed


# --------------------------------------------------------------------------
# Public Entry Point — Dual Engine with Automatic Fallback
# --------------------------------------------------------------------------

def match_resume_to_job(
    resume_text: str,
    job_description: str,
    groq_api_key: str,
    gemini_api_key: Optional[str] = None,
) -> MatchResult:
    """
    Runs resume-to-job matching using Groq as the primary engine.
    Automatically falls back to Gemini if Groq fails, rate-limits, or errors server-side.

    Args:
        resume_text (str): Extracted resume text.
        job_description (str): Extracted/pasted job description text.
        groq_api_key (str): User-provided Groq API key.
        gemini_api_key (Optional[str]): User-provided Gemini API key, used only on fallback.

    Returns:
        MatchResult: Validated structured match result.

    Raises:
        MatcherError: If both engines fail, or if no valid API key is available for a working engine.
    """
    if not resume_text or not resume_text.strip():
        raise MatcherError("Resume text is empty. Cannot run matching.")
    if not job_description or not job_description.strip():
        raise MatcherError("Job description text is empty. Cannot run matching.")

    groq_error_summary = None

    # ---- Attempt 1: Groq (Primary) ----
    if groq_api_key and groq_api_key.strip():
        try:
            logger.info("Attempting match via Groq (llama-3.3-70b-versatile)...")
            parsed = _call_groq(groq_api_key.strip(), resume_text, job_description)
            result = MatchResult(**parsed, engine_used="groq")
            logger.info("Groq match succeeded.")
            return result

        except (GroqRateLimitError, GroqAPIStatusError) as e:
            status_code = getattr(e, "status_code", None)
            logger.warning(f"Groq API error (status={status_code}): {e}. Falling back to Gemini.")
            groq_error_summary = f"Groq API error (status={status_code}): {str(e)}"

        except GroqAPIConnectionError as e:
            logger.warning(f"Groq connection error: {e}. Falling back to Gemini.")
            groq_error_summary = f"Groq connection error: {str(e)}"

        except (ValueError, json.JSONDecodeError) as e:
            logger.warning(f"Groq returned malformed JSON: {e}. Falling back to Gemini.")
            groq_error_summary = f"Groq returned malformed JSON: {str(e)}"

        except ValidationError as e:
            logger.warning(f"Groq output failed schema validation: {e}. Falling back to Gemini.")
            groq_error_summary = f"Groq output failed schema validation: {str(e)}"

        except Exception as e:
            logger.warning(f"Unexpected Groq failure: {e}. Falling back to Gemini.")
            groq_error_summary = f"Unexpected Groq failure: {str(e)}"
    else:
        logger.info("No Groq API key provided. Skipping to Gemini.")
        groq_error_summary = "No Groq API key provided."

    # ---- Attempt 2: Gemini (Fallback) ----
    if gemini_api_key and gemini_api_key.strip():
        try:
            logger.info("Attempting match via Gemini (gemini-2.5-flash) fallback...")
            parsed = _call_gemini(gemini_api_key.strip(), resume_text, job_description)
            result = MatchResult(**parsed, engine_used="gemini")
            logger.info("Gemini fallback match succeeded.")
            return result

        except GeminiAPIError as e:
            logger.error(f"Gemini API error: {e}")
            raise MatcherError(
                f"Both engines failed.\nGroq: {groq_error_summary}\nGemini API error: {str(e)}"
            ) from e

        except (ValueError, json.JSONDecodeError) as e:
            logger.error(f"Gemini returned malformed JSON: {e}")
            raise MatcherError(
                f"Both engines failed.\nGroq: {groq_error_summary}\nGemini returned malformed JSON: {str(e)}"
            ) from e

        except ValidationError as e:
            logger.error(f"Gemini output failed schema validation: {e}")
            raise MatcherError(
                f"Both engines failed.\nGroq: {groq_error_summary}\nGemini output failed schema validation: {str(e)}"
            ) from e

        except Exception as e:
            logger.error(f"Unexpected Gemini failure: {e}")
            raise MatcherError(
                f"Both engines failed.\nGroq: {groq_error_summary}\nUnexpected Gemini failure: {str(e)}"
            ) from e
    else:
        raise MatcherError(
            f"Groq failed and no Gemini API key was provided as fallback.\nGroq: {groq_error_summary}"
        )


# --------------------------------------------------------------------------
# Cover Letter Generation (used by app.py)
# --------------------------------------------------------------------------

COVER_LETTER_SYSTEM_PROMPT = (
    "You are an expert career coach and professional writer. "
    "You write concise, compelling, personalized cover letters based on a candidate's "
    "resume and a target job description. Respond with plain text only — no markdown, "
    "no JSON, no headers or labels. Just the cover letter body text."
)


def _build_cover_letter_prompt(resume_text: str, job_description: str, tone: str) -> str:
    return (
        f"Write a {tone} cover letter (max 350 words) for this candidate applying to this role. "
        f"Highlight the candidate's most relevant strengths and address the role's key requirements. "
        f"Do not fabricate experience not present in the resume.\n\n"
        f"--- RESUME ---\n{resume_text.strip()}\n\n"
        f"--- JOB DESCRIPTION ---\n{job_description.strip()}\n"
    )


def generate_cover_letter(
    resume_text: str,
    job_description: str,
    groq_api_key: str,
    gemini_api_key: Optional[str] = None,
    tone: str = "professional",
) -> str:
    """
    Generates a personalized cover letter using Groq primary / Gemini fallback logic,
    mirroring the same dual-engine reliability pattern as match_resume_to_job().

    Args:
        resume_text (str): Extracted resume text.
        job_description (str): Extracted/pasted job description text.
        groq_api_key (str): User-provided Groq API key.
        gemini_api_key (Optional[str]): User-provided Gemini API key, used only on fallback.
        tone (str): Desired tone, e.g. "professional", "enthusiastic", "concise".

    Returns:
        str: The generated cover letter text.

    Raises:
        MatcherError: If both engines fail.
    """
    if not resume_text or not resume_text.strip():
        raise MatcherError("Resume text is empty. Cannot generate cover letter.")
    if not job_description or not job_description.strip():
        raise MatcherError("Job description text is empty. Cannot generate cover letter.")

    prompt = _build_cover_letter_prompt(resume_text, job_description, tone)
    groq_error_summary = None

    # ---- Attempt 1: Groq (Primary) ----
    if groq_api_key and groq_api_key.strip():
        try:
            logger.info("Generating cover letter via Groq...")
            client = Groq(api_key=groq_api_key.strip())
            completion = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": COVER_LETTER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.6,
                max_tokens=1024,
            )
            text = completion.choices[0].message.content
            if text and text.strip():
                return text.strip()
            raise ValueError("Groq returned empty cover letter text.")

        except (GroqRateLimitError, GroqAPIStatusError) as e:
            status_code = getattr(e, "status_code", None)
            logger.warning(f"Groq cover letter error (status={status_code}): {e}. Falling back to Gemini.")
            groq_error_summary = f"Groq API error (status={status_code}): {str(e)}"

        except GroqAPIConnectionError as e:
            logger.warning(f"Groq connection error: {e}. Falling back to Gemini.")
            groq_error_summary = f"Groq connection error: {str(e)}"

        except Exception as e:
            logger.warning(f"Unexpected Groq failure: {e}. Falling back to Gemini.")
            groq_error_summary = f"Unexpected Groq failure: {str(e)}"
    else:
        groq_error_summary = "No Groq API key provided."

    # ---- Attempt 2: Gemini (Fallback) ----
    if gemini_api_key and gemini_api_key.strip():
        try:
            logger.info("Generating cover letter via Gemini fallback...")
            client = genai.Client(api_key=gemini_api_key.strip())
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=COVER_LETTER_SYSTEM_PROMPT,
                    temperature=0.6,
                    max_output_tokens=1024,
                ),
            )
            if response.text and response.text.strip():
                return response.text.strip()
            raise ValueError("Gemini returned empty cover letter text.")

        except Exception as e:
            logger.error(f"Gemini cover letter fallback failed: {e}")
            raise MatcherError(
                f"Both engines failed to generate a cover letter.\n"
                f"Groq: {groq_error_summary}\nGemini: {str(e)}"
            ) from e
    else:
        raise MatcherError(
            f"Groq failed and no Gemini API key was provided as fallback.\nGroq: {groq_error_summary}"
        )
