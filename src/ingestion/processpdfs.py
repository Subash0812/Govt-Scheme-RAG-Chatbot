from pathlib import Path

import fitz

from cleaner import clean_document


INPUT_DIR = Path("data/raw")

OUTPUT_DIR = Path(
    "data/processed/cleaned_v2"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def extract_text_from_pdf(pdf_path: Path) -> str:

    pages = []

    with fitz.open(pdf_path) as doc:

        for page in doc:

            text = page.get_text("text")

            if text.strip():
                pages.append(text)

    return "\n\n".join(pages)


def process_pdf(pdf_path: Path):

    try:

        print(f"Processing: {pdf_path.name}")

        # PDF → raw text
        raw_text = extract_text_from_pdf(pdf_path)

        if not raw_text.strip():

            print(
                f"WARNING: No text found in {pdf_path.name}"
            )

            return

        # Raw text → cleaned text
        cleaned_text = clean_document(raw_text)

        # Save as TXT
        output_file = (
            OUTPUT_DIR /
            f"{pdf_path.stem}.txt"
        )

        output_file.write_text(
            cleaned_text,
            encoding="utf-8"
        )

        print(
            f"Saved: {output_file.name} | "
            f"Raw: {len(raw_text):,} chars | "
            f"Cleaned: {len(cleaned_text):,} chars"
        )

    except Exception as exc:

        print(
            f"ERROR: {pdf_path.name} -> {exc}"
        )


def main():

    pdf_files = sorted(
        INPUT_DIR.glob("*.pdf")
    )

    print(
        f"Found {len(pdf_files)} PDF files."
    )

    for pdf in pdf_files:
        process_pdf(pdf)

    print("\nCleaning completed.")


if __name__ == "__main__":
    main()