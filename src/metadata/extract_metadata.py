import json
import os
import random
import re
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from google import genai
from pydantic import ValidationError

from src.metadata.metadata_schema import SchemeMetadata


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set.\n"
        "Add it to your .env file."
    )


client = genai.Client(
    api_key=API_KEY
)


# Use the model that worked for your 5-file test.
MODEL_NAME = "gemini-3.6-flash"


INPUT_DIR = Path(
    "data/processed/cleaned_v3"
)

OUTPUT_DIR = Path(
    "data/processed/metadata"
)

FAILED_FILE = OUTPUT_DIR / "failed_metadata.json"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# RATE / RETRY SETTINGS
# ============================================================

# Delay between successful requests.
# This helps avoid hammering the free-tier limits.
REQUEST_DELAY_SECONDS = 4

# Number of retries after a transient error.
MAX_RETRIES = 6

# Initial retry delay.
INITIAL_RETRY_DELAY = 5

# Maximum retry delay.
MAX_RETRY_DELAY = 120


# ============================================================
# SYSTEM INSTRUCTION
# ============================================================

SYSTEM_INSTRUCTION = """
You are a metadata extraction system for an Indian government
scheme RAG dataset.

Your job is EXTRACTION ONLY.

Extract information that is explicitly stated in the document.

DO NOT infer, guess, interpret, or fill missing information from
general knowledge.

============================================================
GENERAL RULES
============================================================

1. Extract ONLY information explicitly present in the document.

2. Do NOT guess missing information.

3. If a field is not explicitly available, return null.

4. Preserve the official scheme name exactly as written.

5. Preserve abbreviations exactly as written.

6. Do not modify:
   - percentages
   - rupee amounts
   - dates
   - URLs
   - department names
   - ministry names
   - organization names

7. Return valid JSON only.

8. Do not summarize the document.

9. Do not rewrite or paraphrase the scheme content.

10. Do not put eligibility conditions into metadata fields unless
    the condition itself explicitly identifies a beneficiary group.

11. The source filename must be preserved exactly.

============================================================
FIELD DEFINITIONS
============================================================

scheme_name:
The official/full name of the scheme explicitly stated in the
document.

abbreviation:
The official abbreviation/acronym explicitly associated with the
scheme, such as AABCS, AABY, A2K+.
If none is explicitly stated, return null.

state:
The Indian State or Union Territory explicitly associated with
the scheme.

Do NOT infer the state from an address unless the document clearly
identifies it as the scheme's state/UT.

ministry:
The Ministry explicitly mentioned as responsible for, implementing,
administering, or sponsoring the scheme.

department:
The Department explicitly mentioned as responsible for,
implementing, administering, or managing the scheme.

scheme_type:
The functional/type classification of the scheme ONLY when
explicitly stated or unambiguously named in the source.

Examples:
- Subsidy
- Insurance
- Scholarship
- Pension
- Loan
- Healthcare
- Training
- Employment
- Social Security

Do not invent a type if it is not supported by the document.

sponsorship:
The sponsorship/funding classification ONLY when explicitly stated.

Possible values include:
- Central Government
- State Government
- Centrally Sponsored
- State Sponsored
- 100% State Sponsored
- Central Sector
- State Sector

IMPORTANT:
Do NOT infer "Central Government" only because a Central Ministry
appears in the document.

If the document explicitly states "100% State Sponsored",
return "100% State Sponsored".

If it explicitly states "Centrally Sponsored Scheme",
return "Centrally Sponsored".

If no sponsorship classification is explicitly stated,
return null.

beneficiary_category:
The classes of people, organizations, institutions, businesses,
or groups that are explicitly identified as beneficiaries or
eligible applicants.

Examples:
- Women
- Farmers
- SC
- ST
- Weavers
- Self-Help Groups
- Small and Micro Enterprises
- Research Institutions

Preserve the wording from the source as much as practical.

Do NOT generalize or infer.

application_mode:
How the scheme/application is explicitly described as being
available.

Examples:
- Online
- Offline
- Online and Offline

official_url:
The official application/scheme/department URL explicitly
present in the document.

Do not invent URLs.

If no URL is present, return null.

============================================================
IMPORTANT DISTINCTION
============================================================

Do NOT confuse:

scheme_type
with
sponsorship

Example:

scheme_type:
"Subsidy"

sponsorship:
"100% State Sponsored"

These are separate fields.

============================================================
REQUIRED JSON FORMAT
============================================================

{
    "scheme_name": string | null,
    "abbreviation": string | null,
    "state": string | null,
    "ministry": string | null,
    "department": string | null,
    "scheme_type": string | null,
    "sponsorship": string | null,
    "beneficiary_category": [],
    "application_mode": string | null,
    "official_url": string | null
}

Return JSON only.
"""


# ============================================================
# TEXT PREPARATION
# ============================================================

def normalize_for_llm(text: str) -> str:
    """
    Remove only obvious transport/whitespace noise.
    Do not rewrite the document.
    """

    text = text.replace("\x00", "")

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


def build_llm_input(
    text: str,
    source_file: str
) -> str:

    text = normalize_for_llm(text)

    # Keep request size bounded.
    max_chars = 12000

    if len(text) <= max_chars:

        selected_text = text

    else:

        first_part = text[:9000]

        last_part = text[-3000:]

        selected_text = (
            first_part
            + "\n\n"
            + "[END OF DOCUMENT EXCERPT]\n\n"
            + last_part
        )

    return f"""
SOURCE FILE:
{source_file}

DOCUMENT TEXT:
--------------------
{selected_text}
--------------------

Extract the metadata fields specified in the system instruction.

Return JSON only.
"""


# ============================================================
# LOAD FAILED FILES
# ============================================================

def load_failed_records() -> list:

    if not FAILED_FILE.exists():

        return []

    try:

        with FAILED_FILE.open(
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

            if isinstance(data, list):
                return data

    except Exception as exc:

        print(
            f"WARNING: Could not read {FAILED_FILE}: {exc}"
        )

    return []


def save_failed_records(records: list) -> None:

    FAILED_FILE.write_text(
        json.dumps(
            records,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )


# ============================================================
# GEMINI CALL WITH RETRIES
# ============================================================

def extract_with_gemini(
    text: str,
    source_file: str
) -> dict:

    prompt = build_llm_input(
        text=text,
        source_file=source_file
    )

    last_exception = None

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config={
                    "system_instruction":
                        SYSTEM_INSTRUCTION,

                    "temperature": 0,

                    "response_mime_type":
                        "application/json",
                },
            )

            raw_response = (
                response.text.strip()
            )

            if not raw_response:

                raise ValueError(
                    "Gemini returned an empty response."
                )

            try:

                return json.loads(
                    raw_response
                )

            except json.JSONDecodeError as exc:

                raise ValueError(
                    "Gemini returned invalid JSON."
                ) from exc

        except Exception as exc:

            last_exception = exc

            error_text = str(exc).lower()

            # Transient conditions that should be retried.
            transient = (
                "503" in error_text
                or "unavailable" in error_text
                or "429" in error_text
                or "resource_exhausted" in error_text
                or "500" in error_text
                or "internal" in error_text
                or "timeout" in error_text
                or "temporarily" in error_text
            )

            if not transient:

                raise

            if attempt >= MAX_RETRIES:

                break

            # Exponential backoff + jitter.
            delay = min(
                INITIAL_RETRY_DELAY
                * (2 ** (attempt - 1)),
                MAX_RETRY_DELAY
            )

            jitter = random.uniform(
                0,
                2
            )

            total_delay = (
                delay + jitter
            )

            print(
                f"  Transient Gemini error "
                f"on attempt {attempt}/{MAX_RETRIES}."
            )

            print(
                f"  Retrying in "
                f"{total_delay:.1f} seconds..."
            )

            time.sleep(
                total_delay
            )

    raise RuntimeError(
        f"Gemini failed after "
        f"{MAX_RETRIES} attempts: "
        f"{last_exception}"
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_metadata(
    data: dict,
    source_file: str
) -> SchemeMetadata:

    # We control this field.
    data["source_file"] = source_file

    try:

        return SchemeMetadata(
            **data
        )

    except ValidationError as exc:

        raise ValueError(
            f"Metadata validation failed: {exc}"
        ) from exc


# ============================================================
# URL VALIDATION
# ============================================================

def validate_url(
    metadata: SchemeMetadata
) -> None:

    if not metadata.official_url:
        return

    url = metadata.official_url.lower()

    if not (
        url.startswith("http://")
        or url.startswith("https://")
    ):

        print(
            f"WARNING: Suspicious URL in "
            f"{metadata.source_file}: "
            f"{metadata.official_url}"
        )


# ============================================================
# PROCESS ONE FILE
# ============================================================

def process_file(
    file_path: Path,
    failed_records: list
) -> bool:

    source_file = file_path.name

    output_path = (
        OUTPUT_DIR
        / f"{file_path.stem}.json"
    )

    # --------------------------------------------------------
    # Resume capability:
    # if metadata already exists, don't call Gemini again.
    # --------------------------------------------------------

    if output_path.exists():

        print(
            f"SKIP: {source_file} "
            f"(metadata already exists)"
        )

        return True

    print(
        f"\nProcessing: {source_file}"
    )

    try:

        text = file_path.read_text(
            encoding="utf-8"
        )

        if not text.strip():

            raise ValueError(
                "Document is empty."
            )

        raw_metadata = extract_with_gemini(
            text=text,
            source_file=source_file
        )

        metadata = validate_metadata(
            raw_metadata,
            source_file
        )

        validate_url(
            metadata
        )

        output_path.write_text(
            metadata.model_dump_json(
                indent=2,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )

        print(
            "  SUCCESS"
        )

        print(
            f"  Scheme: "
            f"{metadata.scheme_name}"
        )

        print(
            f"  State: "
            f"{metadata.state}"
        )

        print(
            f"  Type: "
            f"{metadata.scheme_type}"
        )

        print(
            f"  Saved: "
            f"{output_path}"
        )

        return True

    except Exception as exc:

        print(
            f"  ERROR: {exc}"
        )

        failed_records.append(
            {
                "source_file":
                    source_file,

                "error":
                    str(exc),

                "timestamp":
                    time.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
            }
        )

        save_failed_records(
            failed_records
        )

        return False


# ============================================================
# MAIN
# ============================================================

def main():

    files = sorted(
        INPUT_DIR.glob("*.txt")
    )

    print(
        f"Found {len(files)} cleaned TXT files."
    )

    if not files:

        print(
            "No TXT files found."
        )

        return

    failed_records = load_failed_records()

    success_count = 0
    skipped_count = 0
    failure_count = 0

    existing_files = set(
        p.name
        for p in OUTPUT_DIR.glob("*.json")
    )

    for index, file_path in enumerate(
        files,
        start=1
    ):

        output_name = (
            f"{file_path.stem}.json"
        )

        if output_name in existing_files:

            print(
                f"[{index}/{len(files)}] "
                f"SKIP: {file_path.name}"
            )

            skipped_count += 1

            continue

        print(
            f"\n[{index}/{len(files)}]"
        )

        success = process_file(
            file_path,
            failed_records
        )

        if success:

            success_count += 1

        else:

            failure_count += 1

        # Delay AFTER every attempted API request.
        time.sleep(
            REQUEST_DELAY_SECONDS
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "METADATA EXTRACTION COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        f"Total files     : {len(files)}"
    )

    print(
        f"New successful  : {success_count}"
    )

    print(
        f"Already existed : {skipped_count}"
    )

    print(
        f"Failed          : {failure_count}"
    )

    if failure_count:

        print(
            f"\nFailed files saved to:"
        )

        print(
            f"{FAILED_FILE}"
        )

    print(
        "\nYou can safely rerun the script."
    )

    print(
        "Existing JSON metadata will be skipped."
    )


if __name__ == "__main__":
    main()