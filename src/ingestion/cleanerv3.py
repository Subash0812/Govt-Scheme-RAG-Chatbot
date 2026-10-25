import re
import unicodedata
from typing import List, Optional

import ftfy


# ============================================================
# SECTION HEADINGS USED BY THE MYSCHEME DATASET
# ============================================================

SECTION_HEADINGS = [
    "Frequently Asked Questions",
    "Documents Required",
    "Application Process",
    "Sources And References",
    "Supported Programmes",
    "Objective",
    "Objectives",
    "Benefits",
    "Eligibility",
    "Exclusions",
    "Introduction",
    "Details",
]


# Longer headings must be checked first.
SECTION_HEADINGS = sorted(
    SECTION_HEADINGS,
    key=len,
    reverse=True,
)


# ============================================================
# 1. ENCODING
# ============================================================

def fix_encoding(text: str) -> str:
    """
    Fix common PDF extraction encoding problems.

    Examples:
        â‚¹       -> ₹
        ï»¿       -> BOM removed
        beneï¬ts  -> benefits
    """

    text = ftfy.fix_text(text)
    text = text.replace("\ufeff", "")
    text = unicodedata.normalize("NFKC", text)

    return text


# ============================================================
# 2. BASIC LINE CLEANUP
# ============================================================

def normalize_line(line: str) -> str:

    line = line.replace("\r", " ")
    line = line.replace("\t", " ")

    # Remove excessive spaces
    line = re.sub(r"[ \t]+", " ", line)

    return line.strip()


# ============================================================
# 3. MYSCHEME UI / WEBSITE NOISE
# ============================================================

def is_ui_noise(text: str) -> bool:

    value = text.strip()

    if not value:
        return False

    patterns = [
        r"^Are you sure you want to sign out\??$",
        r"^Cancel$",
        r"^Sign Out$",
        r"^Sign In$",
        r"^Back$",
        r"^Feedback$",
        r"^Share$",
        r"^Was this helpful\??$",
        r"^News and Updates$",
        r"^No new news and updates available$",
        r"^Something went wrong.*$",
        r"^Please try again later.*$",
        r"^You need to.*apply.*schemes.*$",
        r"^It seems you have already initiated.*$",
        r"^To know more please visit.*$",
        r"^Apply Now$",
        r"^Check Eligibility$",
        r"^Ok$",
        r"^Feed Ok$",
        r"^English/हिंदी.*$",
        r"^EngEnglish/हिंदी.*$",
    ]

    for pattern in patterns:
        if re.match(pattern, value, flags=re.IGNORECASE):
            return True

    # Footer fragments
    footer_patterns = [
        "Powered by Digital India Corporation",
        "Ministry of Electronics & IT",
        "support-myscheme",
        "Screen Reader",
        "Accessibility Statement",
        "Disclaimer",
        "Quick Links",
        "Useful Links",
        "Get in touch",
        "Last Updated On",
    ]

    lower = value.lower()

    if any(pattern.lower() in lower for pattern in footer_patterns):
        return True

    return False


# ============================================================
# 4. NAVIGATION STRINGS
# ============================================================

def is_navigation(text: str) -> bool:

    value = re.sub(r"\s+", " ", text.strip()).lower()

    if not value:
        return False

    navigation_words = {
        "details",
        "benefits",
        "eligibility",
        "exclusions",
        "application process",
        "documents required",
        "frequently asked questions",
        "sources and references",
    }

    # Split common concatenated navigation menu
    parts = re.split(
        r"(details|benefits|eligibility|exclusions|application process|"
        r"documents required|frequently asked questions|sources and references)",
        value,
        flags=re.IGNORECASE,
    )

    words = [
        p.strip()
        for p in parts
        if p.strip()
    ]

    if words and all(word in navigation_words for word in words):
        return True

    return value in navigation_words


# ============================================================
# 5. WEBSITE CATEGORY / TAG LINE
# ============================================================

def is_category_line(text: str) -> bool:

    value = text.strip()

    # Example:
    # SchemeCapitalEntrepreneurInvestmentSubsidyUnemployed Youth
    #
    # These are metadata tags, not useful document content.

    tag_patterns = [
        r"^Scheme[A-Za-z]+.*$",
        r"^[A-Za-z]+Scheme[A-Za-z]+.*$",
    ]

    for pattern in tag_patterns:

        if re.match(
            pattern,
            value,
            flags=re.IGNORECASE
        ):

            # Avoid accidentally deleting normal sentences.
            if len(value.split()) <= 12:
                return True

    return False


# ============================================================
# 6. REMOVE PDF-INSERTED PAGE MARKERS
# ============================================================

def remove_page_markers(text: str) -> str:

    return re.sub(
        r"<PARSED TEXT FOR PAGE:\s*\d+\s*/\s*\d+>",
        "",
        text,
        flags=re.IGNORECASE,
    )


# ============================================================
# 7. FIX BROKEN WORDS
# ============================================================

def fix_hyphenation(text: str) -> str:

    # Example:
    #
    # employ-
    # ment
    #
    # -> employment

    text = re.sub(
        r"(\w)-\s*\n\s*(\w)",
        r"\1\2",
        text,
    )

    return text


# ============================================================
# 8. SECTION HEADING DETECTION
# ============================================================

def split_embedded_heading(text: str) -> str:
    """
    Convert:

        BenefitsUnder the scheme...

    into:

        Benefits

        Under the scheme...

    But do NOT change:

        financial benefits under the scheme

    because the heading must occur at a structural boundary.
    """

    result = text

    for heading in SECTION_HEADINGS:

        # ----------------------------------------------------
        # Case 1:
        #   "BenefitsUnder..."
        #
        # The heading starts the line.
        # ----------------------------------------------------

        pattern = (
            rf"^(?P<head>{re.escape(heading)})"
            rf"(?P<body>[A-Za-z₹0-9\"'(])"
        )

        result = re.sub(
            pattern,
            lambda m: (
                f"{m.group('head')}\n\n"
                f"{m.group('body')}"
            ),
            result,
            flags=re.IGNORECASE,
        )

        # ----------------------------------------------------
        # Case 2:
        #   "...Capital Investment Subsidy.ExclusionsThe..."
        #
        # Only match heading after punctuation or section-like
        # boundaries.
        # ----------------------------------------------------

        pattern = (
            rf"(?P<boundary>[.!?:])"
            rf"\s*"
            rf"(?P<head>{re.escape(heading)})"
            rf"(?P<body>[A-Za-z₹0-9\"'(])"
        )

        result = re.sub(
            pattern,
            lambda m: (
                f"{m.group('boundary')}\n\n"
                f"{m.group('head')}\n\n"
                f"{m.group('body')}"
            ),
            result,
            flags=re.IGNORECASE,
        )

    return result


# ============================================================
# 9. NORMALIZE REAL HEADINGS
# ============================================================

def normalize_heading(line: str) -> str:

    value = line.strip()

    for heading in SECTION_HEADINGS:

        if value.lower() == heading.lower():
            return heading

    return value


# ============================================================
# 10. DETECT DUPLICATED DOCUMENT CONTENT
# ============================================================

def normalized_for_compare(text: str) -> str:

    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(
        r"[^a-z0-9₹]+",
        " ",
        text,
    )

    return text.strip()


def find_duplicate_start(lines: List[str]) -> Optional[int]:
    """
    Detect the second copy of the scheme appearing near
    the end of many MyScheme PDFs.

    Strategy:
      1. Find meaningful opening content.
      2. Search later for a close textual match.
      3. If found, cut the duplicate tail.
    """

    # Build meaningful lines
    normalized = [
        normalized_for_compare(line)
        for line in lines
    ]

    meaningful = [
        (idx, line)
        for idx, line in enumerate(normalized)
        if len(line) >= 80
    ]

    if len(meaningful) < 8:
        return None

    # First few substantial lines form our signature.
    signature = [
        line
        for _, line in meaningful[:6]
    ]

    # Search after the first third of the document.
    search_start = max(
        20,
        len(lines) // 3,
    )

    for start in range(
        search_start,
        len(normalized) - 6,
    ):

        matches = 0

        for j, sig in enumerate(signature):

            candidate = normalized[start + j]

            if not candidate:
                continue

            # Compare first 70 characters.
            left = sig[:70]
            right = candidate[:70]

            if (
                left in right
                or right in left
            ):
                matches += 1

        if matches >= 4:
            return start

    return None


# ============================================================
# 11. REMOVE DUPLICATE PARAGRAPHS
# ============================================================

def remove_duplicate_paragraphs(text: str) -> str:

    paragraphs = re.split(
        r"\n\s*\n",
        text,
    )

    seen = set()
    output = []

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        key = normalized_for_compare(
            paragraph
        )

        # Keep short paragraphs.
        if len(key) < 40:
            output.append(paragraph)
            continue

        if key not in seen:
            seen.add(key)
            output.append(paragraph)

    return "\n\n".join(output)


# ============================================================
# 12. FORMAT LISTS
# ============================================================

def normalize_lists(text: str) -> str:

    # Put numbered steps on separate lines.
    text = re.sub(
        r"\s+(Step\s+\d+\s*:)",
        r"\n\1",
        text,
        flags=re.IGNORECASE,
    )

    # Step 01, Step 02 etc.
    text = re.sub(
        r"\s+(Step\s+\d+\s*:)",
        r"\n\1",
        text,
        flags=re.IGNORECASE,
    )

    # Numbered list items
    text = re.sub(
        r"\s+(\d+\.)\s+",
        r"\n\1 ",
        text,
    )

    # Alphabetic list items A. / B.
    text = re.sub(
        r"\s+([A-Z]\.)\s+",
        r"\n\1 ",
        text,
    )

    # Bullet symbols
    text = re.sub(
        r"\s*•\s*",
        "\n- ",
        text,
    )

    return text


# ============================================================
# 13. WHITESPACE
# ============================================================

def normalize_whitespace(text: str) -> str:

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = re.sub(
        r"[ \t]+\n",
        "\n",
        text,
    )

    text = re.sub(
        r"\n[ \t]+",
        "\n",
        text,
    )

    # Keep at most 2 blank lines.
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    # Remove spaces before punctuation.
    text = re.sub(
        r"\s+([,.;:!?])",
        r"\1",
        text,
    )

    return text.strip()


# ============================================================
# 14. MAIN CLEANER
# ============================================================

def clean_document(raw_text: str) -> str:

    # --------------------------------------------
    # A. Encoding
    # --------------------------------------------

    text = fix_encoding(raw_text)

    # --------------------------------------------
    # B. Page markers
    # --------------------------------------------

    text = remove_page_markers(text)

    # --------------------------------------------
    # C. Hyphenation
    # --------------------------------------------

    text = fix_hyphenation(text)

    # --------------------------------------------
    # D. Break embedded section headings
    # --------------------------------------------

    text = split_embedded_heading(text)

    # --------------------------------------------
    # E. Process lines
    # --------------------------------------------

    raw_lines = text.splitlines()

    cleaned_lines = []

    for raw_line in raw_lines:

        line = normalize_line(raw_line)

        if not line:
            cleaned_lines.append("")
            continue

        if is_ui_noise(line):
            continue

        if is_navigation(line):
            continue

        if is_category_line(line):
            continue

        line = normalize_heading(line)

        cleaned_lines.append(line)

    # --------------------------------------------
    # F. Detect repeated copy of document
    # --------------------------------------------

    duplicate_start = find_duplicate_start(
        cleaned_lines
    )

    if duplicate_start is not None:

        cleaned_lines = cleaned_lines[
            :duplicate_start
        ]

    # --------------------------------------------
    # G. Rebuild
    # --------------------------------------------

    text = "\n".join(
        cleaned_lines
    )

    # --------------------------------------------
    # H. Lists
    # --------------------------------------------

    text = normalize_lists(text)

    # --------------------------------------------
    # I. Remove duplicate paragraphs
    # --------------------------------------------

    text = remove_duplicate_paragraphs(
        text
    )

    # --------------------------------------------
    # J. Final whitespace
    # --------------------------------------------

    text = normalize_whitespace(
        text
    )

    return text