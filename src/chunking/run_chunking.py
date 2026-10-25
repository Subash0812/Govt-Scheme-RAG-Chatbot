import json
from pathlib import Path

from .common import (
    TEXT_DIR,
    OUTPUT_DIR,
    load_metadata,
)

from .fixed_chunker import (
    fixed_size_chunks,
)

from .recursive_chunker import (
    recursive_chunks,
)


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

FIXED_DIR = (
    OUTPUT_DIR / "fixed"
)

RECURSIVE_DIR = (
    OUTPUT_DIR / "recursive"
)

FIXED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RECURSIVE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SAVE JSON
# ============================================================

def save_chunks(
    chunks,
    output_path
):

    output_path.write_text(
        json.dumps(
            chunks,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    files = sorted(
        TEXT_DIR.glob("*.txt")
    )

    print(
        f"Found {len(files)} documents."
    )

    total_fixed = 0
    total_recursive = 0

    for index, text_path in enumerate(
        files,
        start=1
    ):

        print(
            f"\n[{index}/{len(files)}] "
            f"{text_path.name}"
        )

        # -----------------------------------------
        # Load text
        # -----------------------------------------

        text = text_path.read_text(
            encoding="utf-8"
        )

        if not text.strip():
            print("  SKIP: empty text")
            continue

        # -----------------------------------------
        # Load metadata
        # -----------------------------------------

        metadata = load_metadata(
            text_path.name
        )

        # -----------------------------------------
        # Fixed-size
        # -----------------------------------------

        fixed = fixed_size_chunks(
            text=text,
            metadata=metadata
        )

        fixed_output = (
            FIXED_DIR
            / f"{text_path.stem}.json"
        )

        save_chunks(
            fixed,
            fixed_output
        )

        # -----------------------------------------
        # Recursive
        # -----------------------------------------

        recursive = recursive_chunks(
            text=text,
            metadata=metadata
        )

        recursive_output = (
            RECURSIVE_DIR
            / f"{text_path.stem}.json"
        )

        save_chunks(
            recursive,
            recursive_output
        )

        print(
            f"  Fixed chunks     : {len(fixed)}"
        )

        print(
            f"  Recursive chunks : {len(recursive)}"
        )

        total_fixed += len(fixed)
        total_recursive += len(recursive)

    print(
        "\n========================================"
    )

    print(
        "CHUNKING COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        f"Total fixed chunks     : {total_fixed}"
    )

    print(
        f"Total recursive chunks : {total_recursive}"
    )


if __name__ == "__main__":
    main()