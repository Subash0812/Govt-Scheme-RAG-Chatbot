from pathlib import Path
import re
import unicodedata

import fitz
import ftfy


INPUT_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# 1. Extract text from PDF
# ---------------------------------------------------------

def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from all pages of a PDF."""

    doc = fitz.open(pdf_path)

    pages = []

    for page in doc:
        text = page.get_text("text")
        pages.append(text)

    doc.close()

    return "\n\n".join(pages)


# ---------------------------------------------------------
# 2. Fix encoding problems
# ---------------------------------------------------------

def fix_encoding(text: str) -> str:

    text = ftfy.fix_text(text)

    text = text.replace("\ufeff", "")

    text = unicodedata.normalize("NFKC", text)

    return text


# ---------------------------------------------------------
# 3. Remove PDF page markers
# ---------------------------------------------------------

def remove_page_markers(text: str) -> str:

    text = re.sub(
        r"<PARSED TEXT FOR PAGE:\s*\d+\s*/\s*\d+>",
        "",
        text,
        flags=re.IGNORECASE
    )

    return text


# ---------------------------------------------------------
# 4. Remove MyScheme website noise
# ---------------------------------------------------------

def remove_myscheme_noise(text: str):

    patterns = [

        r"Are you sure you want to sign out\?",
        r"Cancel",
        r"Sign Out",
        r"Sign In",
        r"Back",
        r"English/हिन्दी",

        r"Something went wrong\.?",
        r"Please try again later\.?",

        r"You need to sign in before applying for schemes",
        r"It seems you have already initiated your application earlier",
        r"To know more please visit",

        r"Apply Now",
        r"Check Eligibility",

        r"Was this helpful\?",
        r"News and Updates",
        r"No new news and updates available",
        r"Share",
        r"Feedback",

        r"Quick Links",
        r"Screen Reader",
        r"Accessibility Statement",
        r"Frequently Asked Questions",
        r"Disclaimer",
        r"Useful Links",
        r"Get in touch",

        r"©2024.*?Last Updated On.*",
    ]

    for pattern in patterns:
        text = re.sub(
            pattern,
            " ",
            text,
            flags=re.IGNORECASE | re.DOTALL
        )

    return text


# ---------------------------------------------------------
# 5. Remove navigation
# ---------------------------------------------------------

def remove_navigation(text: str):

    patterns = [
        r"Details\s+Benefits\s+Eligibility\s+Exclusions\s+Application Process\s+Documents Required\s+Frequently Asked Questions",
        r"Details\s+Benefits\s+Eligibility\s+Application Process\s+Documents Required\s+Frequently Asked Questions",
        r"Benefits\s+Eligibility\s+Application Process\s+Documents Required\s+Frequently Asked Questions",
        r"Sources\s+And\s+References",
    ]

    for pattern in patterns:
        text = re.sub(
            pattern,
            " ",
            text,
            flags=re.IGNORECASE
        )

    return text


# ---------------------------------------------------------
# 6. Fix broken words
# ---------------------------------------------------------

def fix_hyphenation(text: str):

    # Example:
    # indus-
    # trial
    #
    # becomes:
    # industrial

    text = re.sub(
        r"(\w)-\s*\n\s*(\w)",
        r"\1\2",
        text
    )

    return text


# ---------------------------------------------------------
# 7. Restore section headings
# ---------------------------------------------------------

SECTION_HEADINGS = [
    "Benefits",
    "Eligibility",
    "Exclusions",
    "Application Process",
    "Documents Required",
    "Frequently Asked Questions",
    "Sources And References",
    "Objective",
    "Objectives",
    "Introduction",
    "Details",
    "Supported Programmes",
]


def restore_section_headings(text: str):

    for heading in SECTION_HEADINGS:

        pattern = rf"\s*{re.escape(heading)}\s*"

        text = re.sub(
            pattern,
            f"\n\n{heading}\n\n",
            text,
            flags=re.IGNORECASE
        )

    return text


# ---------------------------------------------------------
# 8. Normalize whitespace
# ---------------------------------------------------------

def normalize_whitespace(text: str):

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)

    # maximum 2 consecutive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # multiple spaces
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


# ---------------------------------------------------------
# 9. Main cleaning function
# ---------------------------------------------------------

def clean_text(text: str) -> str:

    text = fix_encoding(text)

    text = remove_page_markers(text)

    text = remove_myscheme_noise(text)

    text = remove_navigation(text)

    text = fix_hyphenation(text)

    text = restore_section_headings(text)

    text = normalize_whitespace(text)

    return text


# ---------------------------------------------------------
# 10. Process PDFs
# ---------------------------------------------------------

for pdf_path in INPUT_DIR.glob("*.pdf"):

    print(f"Processing: {pdf_path.name}")

    try:

        # PDF → raw text
        raw_text = extract_pdf_text(pdf_path)

        # Raw text → cleaned text
        cleaned_text = clean_text(raw_text)

        # Save as TXT
        output_path = OUTPUT_DIR / f"{pdf_path.stem}.txt"

        output_path.write_text(
            cleaned_text,
            encoding="utf-8"
        )

        print(
            f"Saved: {output_path.name} | "
            f"{len(raw_text):,} raw chars → "
            f"{len(cleaned_text):,} cleaned chars"
        )

    except Exception as e:

        print(
            f"ERROR processing {pdf_path.name}: {e}"
        )