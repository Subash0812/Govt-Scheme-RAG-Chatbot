import re
import unicodedata
from typing import List

import ftfy


# =========================================================
# SECTION HEADINGS THAT WE ACTUALLY EXPECT FROM MYSCHEME
# =========================================================

SECTION_HEADINGS = {
    "Details",
    "Introduction",
    "Objective",
    "Objectives",
    "Supported Programmes",
    "Benefits",
    "Eligibility",
    "Exclusions",
    "Application Process",
    "Documents Required",
    "Frequently Asked Questions",
    "Sources And References",
}


# =========================================================
# 1. ENCODING CLEANUP
# =========================================================

def fix_encoding(text: str) -> str:
    """
    Fix common PDF text encoding problems.
    """

    text = ftfy.fix_text(text)

    # Remove BOM
    text = text.replace("\ufeff", "")

    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)

    return text


# =========================================================
# 2. BASIC PDF EXTRACTION CLEANUP
# =========================================================

def clean_line(line: str) -> str:
    """
    Basic cleanup for a single extracted line.
    """

    line = line.replace("\r", " ")
    line = line.replace("\t", " ")

    # Remove repeated spaces
    line = re.sub(r"[ \t]+", " ", line)

    return line.strip()


# =========================================================
# 3. REMOVE CLEAR MYScheme UI NOISE
# =========================================================

NOISE_EXACT_LINES = {
    "EngEnglish/हिंदीSign In",
    "English/हिंदीSign In",
    "Sign In",
    "Sign Out",
    "Back",
    "Feed Ok",
    "Feedback",
    "Share",
    "Was this helpful?",
    "News and Updates",
    "No new news and updates available",
    "GuidelinesOfficial WebsiteOk",
}


def is_noise_line(line: str) -> bool:

    stripped = line.strip()

    if not stripped:
        return False

    # Exact known UI text
    if stripped in NOISE_EXACT_LINES:
        return True

    # Generic UI fragments seen in these PDFs
    noise_patterns = [
        r"^Are you sure you want to sign out\??$",
        r"^Cancel$",
        r"^Apply Now$",
        r"^Check Eligibility$",
        r"^Something went wrong.*$",
        r"^Please try again later.*$",
        r"^You need to.*apply.*schemes.*$",
        r"^It seems you have already initiated.*$",
        r"^To know more please visit.*$",
        r"^Ok$",
        r"^\. Check$",
        r"^No new.*available.*$",
        r"^Quick Links$",
        r"^Useful Links$",
        r"^Get in touch$",
        r"^Screen Reader$",
        r"^Accessibility Statement$",
        r"^Disclaimer$",
    ]

    for pattern in noise_patterns:
        if re.match(pattern, stripped, flags=re.IGNORECASE):
            return True

    # Footer
    if "Powered by Digital India Corporation" in stripped:
        return True

    if "Ministry of Electronics & IT" in stripped:
        return True

    if "support-myscheme" in stripped.lower():
        return True

    if "Last Updated On" in stripped:
        return True

    return False


# =========================================================
# 4. REMOVE NAVIGATION-ONLY LINES
# =========================================================

NAVIGATION_PATTERN = re.compile(
    r"^(Details|Benefits|Eligibility|Exclusions|Application Process|"
    r"Documents Required|Frequently Asked Questions|Sources And References)"
    r"(\s+(Details|Benefits|Eligibility|Exclusions|Application Process|"
    r"Documents Required|Frequently Asked Questions|Sources And References))*$",
    re.IGNORECASE
)


def is_navigation_line(line: str) -> bool:

    stripped = line.strip()

    if not stripped:
        return False

    return bool(NAVIGATION_PATTERN.match(stripped))


# =========================================================
# 5. REMOVE COMMON WEBSITE TITLE/TAG NOISE
# =========================================================

def clean_scheme_header(line: str) -> str:
    """
    Clean lines such as:

    Lakshadweep25% Capital Investment Subsidy
    SchemeCapitalEntrepreneurInvestmentSubsidyUnemployed Youth

    We preserve the meaningful title and remove obvious tags.
    """

    # Remove common concatenated category/tag line
    if re.search(
        r"SchemeCapital|SchemeHandloom|SchemeHealth|SchemeIndustry|"
        r"SchemeInstitution|SchemeResearch",
        line,
        flags=re.IGNORECASE,
    ):
        return ""

    # Remove leading state/UT + title concatenation only when obvious.
    # We do NOT aggressively modify ordinary content.
    return line


# =========================================================
# 6. FIX BROKEN HYPHENATION
# =========================================================

def fix_hyphenation(text: str) -> str:
    """
    Example:

        self-
        employment

    becomes:

        selfemployment

    Then we fix it only when the extracted line break
    clearly represents a word split.
    """

    text = re.sub(
        r"(\w)-\s*\n\s*(\w)",
        r"\1\2",
        text
    )

    return text


# =========================================================
# 7. REMOVE REPEATED DOCUMENT TAIL
# =========================================================

def find_repeated_tail(lines: List[str]) -> int | None:
    """
    MyScheme PDFs often contain the actual scheme content,
    then website/UI content, followed by a repeated copy
    of the beginning of the scheme.

    We detect a repeated long sequence by comparing the
    beginning with later lines.

    Returns the line index where the repeated copy starts.
    """

    # Build normalized lines
    normalized = []

    for line in lines:
        value = re.sub(r"\s+", " ", line.strip().lower())
        normalized.append(value)

    # Need enough content
    if len(normalized) < 40:
        return None

    # Use the first 6 meaningful lines as a signature,
    # ignoring generic headers.
    first_candidates = []

    for value in normalized:
        if len(value) >= 50:
            first_candidates.append(value)

        if len(first_candidates) == 6:
            break

    if len(first_candidates) < 3:
        return None

    # Search later in the document
    for i in range(20, len(normalized) - len(first_candidates)):

        matches = 0

        for j, signature in enumerate(first_candidates):
            candidate = normalized[i + j]

            if not candidate:
                continue

            # Compare beginnings to tolerate line wrapping differences
            if (
                candidate.startswith(signature[:60])
                or signature.startswith(candidate[:60])
            ):
                matches += 1

        # 4 out of 6 matching lines is strong evidence of
        # duplicated document content.
        if matches >= 4:
            return i

    return None


# =========================================================
# 8. REMOVE DUPLICATE CONSECUTIVE PARAGRAPHS
# =========================================================

def remove_duplicate_paragraphs(text: str) -> str:

    paragraphs = re.split(r"\n\s*\n", text)

    seen = set()
    result = []

    for paragraph in paragraphs:

        normalized = re.sub(
            r"\s+",
            " ",
            paragraph.strip().lower()
        )

        if len(normalized) < 30:
            result.append(paragraph.strip())
            continue

        if normalized in seen:
            continue

        seen.add(normalized)
        result.append(paragraph.strip())

    return "\n\n".join(
        p for p in result if p
    )


# =========================================================
# 9. RESTORE ONLY REAL SECTION HEADINGS
# =========================================================

def normalize_heading_line(line: str) -> str:
    """
    IMPORTANT:
    This only acts on lines that already look like headings.

    It does NOT search and replace every occurrence of
    "Benefits" or "Eligibility" inside ordinary sentences.
    """

    stripped = line.strip()

    for heading in SECTION_HEADINGS:

        if stripped.lower() == heading.lower():
            return heading

    return stripped


# =========================================================
# 10. PARAGRAPH CLEANUP
# =========================================================

def normalize_text_layout(text: str) -> str:

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Clean spaces around lines
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)

    # Excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# =========================================================
# 11. MAIN CLEANING FUNCTION
# =========================================================

def clean_document(raw_text: str) -> str:

    # -----------------------------------------------
    # A. Encoding
    # -----------------------------------------------

    text = fix_encoding(raw_text)

    # -----------------------------------------------
    # B. Fix word splits before line processing
    # -----------------------------------------------

    text = fix_hyphenation(text)

    # -----------------------------------------------
    # C. Split into lines
    # -----------------------------------------------

    raw_lines = text.splitlines()

    cleaned_lines = []

    for raw_line in raw_lines:

        line = clean_line(raw_line)

        if not line:
            cleaned_lines.append("")
            continue

        # Website/UI noise
        if is_noise_line(line):
            continue

        # Navigation-only lines
        if is_navigation_line(line):
            continue

        # Header/tag cleanup
        line = clean_scheme_header(line)

        if not line:
            continue

        # Normalize genuine heading lines
        line = normalize_heading_line(line)

        cleaned_lines.append(line)

    # -----------------------------------------------
    # D. Remove duplicate tail
    # -----------------------------------------------

    duplicate_start = find_repeated_tail(cleaned_lines)

    if duplicate_start is not None:

        cleaned_lines = cleaned_lines[:duplicate_start]

    # -----------------------------------------------
    # E. Rebuild text
    # -----------------------------------------------

    text = "\n".join(cleaned_lines)

    # -----------------------------------------------
    # F. Remove duplicate paragraphs
    # -----------------------------------------------

    text = remove_duplicate_paragraphs(text)

    # -----------------------------------------------
    # G. Final layout cleanup
    # -----------------------------------------------

    text = normalize_text_layout(text)

    return text