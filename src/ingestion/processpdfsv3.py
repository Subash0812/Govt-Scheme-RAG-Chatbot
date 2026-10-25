from pathlib import Path

import fitz

from cleanerv3 import clean_document


# ============================================================
# PATHS
# ============================================================

INPUT_DIR = Path(
    "data/raw_filtered"
)

OUTPUT_DIR = Path(
    "data/processed/cleaned_v3"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf_text(pdf_path: Path) -> str:
    """
    Extract text from every page.
    """

    pages = []

    try:

        with fitz.open(pdf_path) as doc:

            for page_number, page in enumerate(doc):

                text = page.get_text(
                    "text"
                )

                if text.strip():
                    pages.append(
                        text
                    )

    except Exception as exc:

        raise RuntimeError(
            f"Could not read PDF: {pdf_path}"
        ) from exc

    return "\n\n".join(
        pages
    )


# ============================================================
# PROCESS ONE PDF
# ============================================================

def process_pdf(pdf_path: Path) -> None:

    print(
        f"\nProcessing: {pdf_path.name}"
    )

    try:

        # 1. PDF → raw text
        raw_text = extract_pdf_text(
            pdf_path
        )

        if not raw_text.strip():

            print(
                "WARNING: PDF contains no extractable text."
            )

            return

        # 2. Raw text → cleaned text
        cleaned_text = clean_document(
            raw_text
        )

        # 3. Save TXT
        output_path = (
            OUTPUT_DIR /
            f"{pdf_path.stem}.txt"
        )

        output_path.write_text(
            cleaned_text,
            encoding="utf-8",
        )

        print(
            f"Raw characters     : {len(raw_text):,}"
        )

        print(
            f"Cleaned characters : {len(cleaned_text):,}"
        )

        print(
            f"Saved              : {output_path}"
        )

    except Exception as exc:

        print(
            f"ERROR: {pdf_path.name}"
        )

        print(
            f"       {exc}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    pdf_files = sorted(
        INPUT_DIR.glob("*.pdf")
    )

    print(
        f"Found {len(pdf_files)} PDF files."
    )

    if not pdf_files:

        print(
            "No PDF files found."
        )

        return

    for pdf_path in pdf_files:

        process_pdf(
            pdf_path
        )

    print(
        "\nV3 cleaning completed."
    )


if __name__ == "__main__":
    main()