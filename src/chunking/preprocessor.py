import re
import unicodedata

import ftfy


# ============================================================
# SECTION HEADINGS
# ============================================================

SECTION_HEADINGS = [
    "Frequently Asked Questions",
    "Documents Required",
    "Application Process",
    "Sources And References",
    "Supported Programmes",
    "Negative List of Activities",
    "Objectives",
    "Objective",
    "Benefits",
    "Eligibility",
    "Exclusions",
    "Introduction",
    "Details",
]

SECTION_HEADINGS = sorted(
    SECTION_HEADINGS,
    key=len,
    reverse=True
)


# ============================================================
# ENCODING
# ============================================================

def fix_encoding(text: str) -> str:

    text = ftfy.fix_text(text)

    text = text.replace("\ufeff", "")

    return unicodedata.normalize("NFKC", text)


# ============================================================
# REMOVE MYScheme HEADER BLOCK
# ============================================================

def remove_myscheme_header(text: str) -> str:
    """
    Remove the MyScheme website/navigation block that appears
    before the actual scheme description.

    We preserve the scheme title and start keeping content from
    the first real descriptive sentence.
    """

    lines = text.splitlines()

    if not lines:
        return text

    # --------------------------------------------------------
    # Find the first strong signal that we are inside the
    # MyScheme UI/header.
    # --------------------------------------------------------

    header_start = None

    ui_signals = [
        "Are you sure you want to sign out",
        "English/हिंदी",
        "EngEnglish/हिंदी",
        "Sign In",
        "Sign Out",
        "BackDetails",
        "Back Details",
        "DetailsBenefits",
        "Details Benefits",
        "You need to",
        "It seems you have already initiated",
        "CancelApply Now",
        "Cancel Apply Now",
        "Check Eligibility",
    ]

    for i, line in enumerate(lines):

        lower = line.lower()

        if any(
            signal.lower() in lower
            for signal in ui_signals
        ):
            header_start = i
            break

    # If no obvious UI exists, don't modify.
    if header_start is None:
        return text

    # --------------------------------------------------------
    # Find the end of the UI region.
    #
    # The actual scheme usually begins around the first
    # meaningful descriptive sentence after the title/state.
    #
    # We use several signals rather than one exact phrase.
    # --------------------------------------------------------

    title_line = lines[0].strip()

    # Keep the first line as title.
    output = [title_line]

    # Search after the UI block for a substantial content line.
    content_start = None

    for i in range(
        max(header_start, 1),
        len(lines)
    ):

        line = lines[i].strip()

        if not line:
            continue

        lower = line.lower()

        # Skip obvious UI remnants.
        if any(
            signal.lower() in lower
            for signal in ui_signals
        ):
            continue

        # Skip navigation-only lines.
        if re.fullmatch(
            r"(Details|Benefits|Eligibility|Exclusions|"
            r"Application Process|Documents Required|"
            r"Frequently Asked Questions|Sources And References)"
            r"(\s+(Details|Benefits|Eligibility|Exclusions|"
            r"Application Process|Documents Required|"
            r"Frequently Asked Questions|Sources And References))*",
            line,
            flags=re.IGNORECASE
        ):
            continue

        # We consider a reasonably long natural-language line
        # to be actual content.
        if len(line) >= 60:

            content_start = i
            break

    if content_start is None:
        return text

    # --------------------------------------------------------
    # Rebuild title + real content.
    # --------------------------------------------------------

    output.append("")

    output.extend(
        lines[content_start:]
    )

    return "\n".join(output)


# ============================================================
# REMOVE UI FRAGMENTS THAT SURVIVE HEADER CLEANING
# ============================================================

def remove_remaining_ui(text: str) -> str:

    patterns = [

        r"EngEnglish/हिंदीSign\s*",
        r"English/हिंदीSign\s*",
        r"CancelSign\s*(In|Out)?",
        r"Sign\s*(In|Out)",
        r"BackDetailsBenefitsEligibility.*?Sources And References",
        r"DetailsBenefitsEligibility.*?Sources And References",

        r"Are you sure you want to sign out\??",
        r"You need to.*?before applying.*?schemes",
        r"It seems you have already initiated your application earlier",
        r"To know more please visit",

        r"Something went wrong.*?",
        r"Please try again later.*?",

        r"Apply Now",
        r"Check Eligibility",

        r"Was this helpful\??",
        r"News and Updates",
        r"No new.*?available",
        r"Share",

        r"Powered by Digital India.*",
        r"©2024.*",
        r"Screen Reader.*",
        r"Accessibility Statement.*",
        r"Disclaimer.*",
        r"Terms & Conditions.*",
        r"Useful Links.*",
        r"Quick Links.*",
        r"Get in touch.*",
    ]

    for pattern in patterns:
        text = re.sub(
            pattern,
            " ",
            text,
            flags=re.IGNORECASE | re.DOTALL
        )

    return text


# ============================================================
# REMOVE CATEGORY / TAG STRINGS
# ============================================================

def remove_category_noise(text: str) -> str:

    patterns = [

        r"SchemeCapitalEntrepreneurInvestmentSubsidyUnemployed Youth",

        r"SchemeHandloomSubsidyWeaver",

        r"IndustryInstitutionProposalResearchTechnology",

        r"IndustryInstitutionProposalResearchTechnologyDetails",
    ]

    for pattern in patterns:

        text = re.sub(
            pattern,
            " ",
            text,
            flags=re.IGNORECASE
        )

    return text


# ============================================================
# SPLIT INLINE SECTION HEADINGS
# ============================================================

def split_inline_headings(text: str) -> str:

    result = text

    for heading in SECTION_HEADINGS:

        # ----------------------------------------------------
        # Heading followed immediately by content.
        #
        # BenefitsUnder...
        # EligibilityAny...
        # ----------------------------------------------------

        pattern = (
            rf"(?<![A-Za-z])"
            rf"({re.escape(heading)})"
            rf"(?=[A-Z₹0-9\"'\-(])"
        )

        result = re.sub(
            pattern,
            r"\n\n\1\n\n",
            result,
            flags=re.IGNORECASE
        )

    return result


# ============================================================
# NORMALIZE LISTS
# ============================================================

def normalize_lists(text: str) -> str:

    text = re.sub(
        r"\s+(Step\s+\d+\s*:)",
        r"\n\1",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s+(\d+\.)\s+",
        r"\n\1 ",
        text
    )

    text = re.sub(
        r"\s*•\s*",
        "\n- ",
        text
    )

    return text


# ============================================================
# WHITESPACE
# ============================================================

def normalize_whitespace(text: str) -> str:

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = re.sub(
        r"[ \t]+\n",
        "\n",
        text
    )

    text = re.sub(
        r"\n[ \t]+",
        "\n",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# ============================================================
# MAIN
# ============================================================

def preprocess_for_chunking(text: str) -> str:

    # 1. Encoding
    text = fix_encoding(text)

    # 2. Remove large header
    text = remove_myscheme_header(text)

    # 3. Remove remaining UI
    text = remove_remaining_ui(text)

    # 4. Remove category/tag noise
    text = remove_category_noise(text)

    # 5. Remove duplicated second copy
    text = find_duplicate_copy(text)

    # 6. Restore section boundaries
    text = split_inline_headings(text)

    # 7. Restore lists
    text = normalize_lists(text)

    # 8. Final whitespace
    text = normalize_whitespace(text)

    return text

# ============================================================
# REMOVE DUPLICATED SECOND COPY
# ============================================================

def normalize_for_match(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9₹%]+", " ", text)
    return text.strip()


def find_duplicate_copy(text: str) -> str:
    """
    MyScheme documents often contain the actual scheme followed
    by a website/footer block and then a second copy of the
    beginning of the same scheme.

    We detect a repeated long opening passage and cut everything
    from that second occurrence onward.
    """

    paragraphs = [
        p.strip()
        for p in re.split(r"\n\s*\n", text)
        if p.strip()
    ]

    if len(paragraphs) < 8:
        return text

    # Use the first substantial content paragraphs as the signature.
    signature = []

    for p in paragraphs:

        normalized = normalize_for_match(p)

        if len(normalized) >= 100:
            signature.append(normalized[:250])

        if len(signature) == 3:
            break

    if len(signature) < 2:
        return text

    # Search later paragraphs for the same content.
    for i in range(3, len(paragraphs) - 2):

        matches = 0

        for j, sig in enumerate(signature):

            candidate = normalize_for_match(
                paragraphs[i + j]
            )

            if not candidate:
                continue

            # Strong similarity based on long prefix.
            if (
                sig[:100] in candidate
                or candidate[:100] in sig
            ):
                matches += 1

        if matches >= 2:

            # Keep everything before the second copy.
            return "\n\n".join(
                paragraphs[:i]
            )

    return text