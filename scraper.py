"""
scraper.py
-----------
Handles two distinct data extraction tasks for the AI Job Scraper & Resume Matcher:

1. Resume Parsing: Extracts raw text from an uploaded PDF resume using pdfplumber.
2. Job Description Scraping: Fetches and parses a job posting URL using requests + BeautifulSoup,
   stripping out noise (scripts, styles, nav, footer) to isolate the core job description text.

Both functions return clean strings ready to be passed into matcher.py for LLM analysis.
"""

import re
import logging
from io import BytesIO
from typing import Optional

import requests
import pdfplumber
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tags that typically contain no useful job-description content
NOISE_TAGS = [
    "script", "style", "nav", "footer", "header",
    "svg", "noscript", "iframe", "form", "button", "aside"
]

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

REQUEST_TIMEOUT = 15  # seconds


class ScraperError(Exception):
    """Raised when scraping or parsing fails in a way the caller should handle."""
    pass


def extract_resume_text(uploaded_file) -> str:
    """
    Extracts text from an uploaded PDF resume.

    Args:
        uploaded_file: A file-like object (e.g., Streamlit's UploadedFile) containing PDF bytes.

    Returns:
        str: Cleaned, concatenated text from all pages of the PDF.

    Raises:
        ScraperError: If the file cannot be read or contains no extractable text.
    """
    try:
        # Streamlit's UploadedFile behaves like BytesIO; ensure compatibility either way
        if hasattr(uploaded_file, "read"):
            file_bytes = uploaded_file.read()
        else:
            file_bytes = uploaded_file

        pdf_stream = BytesIO(file_bytes)
        extracted_pages = []

        with pdfplumber.open(pdf_stream) as pdf:
            if len(pdf.pages) == 0:
                raise ScraperError("The uploaded PDF has no pages.")

            for page_number, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text()
                if page_text:
                    extracted_pages.append(page_text)
                else:
                    logger.warning(f"No extractable text on page {page_number} of resume.")

        full_text = "\n".join(extracted_pages)
        full_text = _clean_text(full_text)

        if not full_text or len(full_text.strip()) < 30:
            raise ScraperError(
                "Could not extract meaningful text from this PDF. "
                "It may be a scanned image without a text layer."
            )

        logger.info(f"Successfully extracted {len(full_text)} characters from resume.")
        return full_text

    except ScraperError:
        raise
    except Exception as e:
        logger.error(f"Failed to parse PDF resume: {e}")
        raise ScraperError(f"Failed to parse PDF resume: {str(e)}") from e


def scrape_job_description(url: str) -> str:
    """
    Fetches a job posting URL and extracts the visible textual content,
    stripping scripts, styles, navigation, and other non-content elements.

    Args:
        url (str): The job posting URL.

    Returns:
        str: Cleaned text representing the job description page content.

    Raises:
        ScraperError: If the URL is invalid, unreachable, or contains no usable content.
    """
    if not url or not url.strip().lower().startswith(("http://", "https://")):
        raise ScraperError("Please provide a valid URL starting with http:// or https://")

    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.exceptions.Timeout as e:
        raise ScraperError("The request to the job URL timed out. Try again or paste the JD manually.") from e
    except requests.exceptions.ConnectionError as e:
        raise ScraperError("Could not connect to the job URL. Check the link and try again.") from e
    except requests.exceptions.HTTPError as e:
        raise ScraperError(f"The job URL returned an error: {e}") from e
    except requests.exceptions.RequestException as e:
        raise ScraperError(f"Failed to fetch job URL: {str(e)}") from e

    try:
        soup = BeautifulSoup(response.text, "html.parser")

        for tag_name in NOISE_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        # Prefer common job-description containers if present, else fall back to <body>
        candidate_selectors = [
            {"class_": re.compile(r"job.?description", re.I)},
            {"class_": re.compile(r"job.?details", re.I)},
            {"id": re.compile(r"job.?description", re.I)},
        ]

        content_node = None
        for selector in candidate_selectors:
            found = soup.find(attrs=selector)
            if found and len(found.get_text(strip=True)) > 100:
                content_node = found
                break

        if content_node is None:
            content_node = soup.body if soup.body else soup

        raw_text = content_node.get_text(separator="\n")
        cleaned = _clean_text(raw_text)

        if not cleaned or len(cleaned.strip()) < 50:
            raise ScraperError(
                "Could not extract meaningful job description text from this page. "
                "Try pasting the job description manually instead."
            )

        logger.info(f"Successfully scraped {len(cleaned)} characters from job URL.")
        return cleaned

    except ScraperError:
        raise
    except Exception as e:
        logger.error(f"Failed to parse job page HTML: {e}")
        raise ScraperError(f"Failed to parse job page content: {str(e)}") from e


def _clean_text(text: str) -> str:
    """
    Normalizes whitespace and removes excessive blank lines from extracted text.

    Args:
        text (str): Raw extracted text.

    Returns:
        str: Cleaned text.
    """
    if not text:
        return ""

    # Collapse multiple spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3+ newlines into 2 (preserve paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespace on each line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(line for line in lines if line != "" or True)  # keep structure

    return text.strip()
