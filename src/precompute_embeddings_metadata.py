import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.encoder_models import create_encoder


# ---------------------------------------------------------------------------
# Metadata representation
# ---------------------------------------------------------------------------

def extract_autocompletion_texts(auto_completions):
    """
    Extract only the textual values from CanLII autoCompletions.

    Example input:

        [
            {
                "type": "DECISION_CANLII_REFERENCE",
                "text": "1986 CanLII 17 (SCC)"
            },
            {
                "type": "MAIN_TITLE",
                "text": "Mills v. The Queen"
            }
        ]

    becomes:

        [
            "1986 CanLII 17 (SCC)",
            "Mills v. The Queen"
        ]

    The completion type is intentionally discarded.
    """

    if not auto_completions:
        return []

    # autoCompletions is itself stored as a JSON string in the CSV.
    if isinstance(auto_completions, str):
        try:
            auto_completions = json.loads(auto_completions)
        except json.JSONDecodeError:
            return []

    if not isinstance(auto_completions, list):
        return []

    texts = []

    for completion in auto_completions:
        if not isinstance(completion, dict):
            continue

        text = completion.get("text")

        if text is not None:
            text = str(text).strip()

            if text:
                texts.append(text)

    return texts


def build_metadata_text(candidate: dict) -> str:
    """
    Build the V1 textual representation used for embedding.

    V1 includes:

        ## - id
        - docType
        - docTitle
        - autoCompletions["text"]

    The autoCompletion type is deliberately NOT encoded.

    Example:

        id: 1986csc-scc38
        docType: DECISION
        docTitle: Mills v. The Queen
        references: 1986 CanLII 17 (SCC) | 1986 CanLII 17 (CSC) |
                    Mills v. The Queen | [1986] 1 SCR 863 | ...
    """

    parts = []

    #candidate_id = candidate.get("id")
    #if candidate_id:
    #    parts.append(str(candidate_id))

    doc_type = candidate.get("docType")
    if doc_type:
        parts.append(str(doc_type))

    doc_title = candidate.get("docTitle")
    if doc_title:
        parts.append(str(doc_title))

    auto_completion_texts = extract_autocompletion_texts(
        candidate.get("autoCompletions")
    )

    if auto_completion_texts:
        parts.extend(auto_completion_texts)

    return " | ".join(parts)


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_candidates(csv_path: Path):
    """
    Load candidate metadata from candidate_pool_metadata.csv.

    Returns:
        candidates:
            List of candidate dictionaries containing the original metadata.

        texts:
            V1 textual representation corresponding to each candidate.
    """

    df = pd.read_csv(csv_path)

    required_columns = {"original_url", "metadata"}

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns in {csv_path}: {sorted(missing)}"
        )

    candidates = []
    texts = []

    for row_idx, row in df.iterrows():
        original_url = row["original_url"]
        metadata_raw = row["metadata"]

        try:
            metadata = json.loads(metadata_raw)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(
                f"Could not parse metadata JSON on row {row_idx}: "
                f"{exc}"
            ) from exc

        if not isinstance(metadata, list):
            raise ValueError(
                f"Expected metadata to be a list on row {row_idx}, "
                f"got {type(metadata).__name__}"
            )

        for candidate in metadata:
            if not isinstance(candidate, dict):
                continue

            text = build_metadata_text(candidate)

            candidates.append(
                {
                    "original_url": original_url,
                    "id": candidate.get("id"),
                    "docType": candidate.get("docType"),
                    "docTitle": candidate.get("docTitle"),
                    "path": candidate.get("path"),
                    "jurisdictionId": candidate.get("jurisdictionId"),
                    "citedCount": candidate.get("citedCount"),
                    "metadata_text_v1": text,
                }
            )

            texts.append(text)

    return candidates, texts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Precompute embeddings for CanLII candidate metadata."
        )
    )

    parser.add_argument(
        "--encoder",
        choices=["bge-s", "splade"],
        default="bge-s",
        help="Embedding model to use.",
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/candidate_pool_metadata.csv"),
        help="Input candidate metadata CSV.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. If omitted, uses "
            "cache/candidate_pool_embeddings_<encoder>_v1."
        ),
    )

    parser.add_argument(
        "--device",
        default=None,
        help="PyTorch device, e.g. cuda or cpu.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Encoding batch size.",
    )

    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Output directory
    # -----------------------------------------------------------------------

    if args.output_dir is None:
        output_dir = Path(
            f"cache/candidate_pool_embeddings_{args.encoder}_v1"
        )
    else:
        output_dir = args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load candidates
    # -----------------------------------------------------------------------

    print(f"Loading candidate metadata from: {args.input}")

    candidates, texts = load_candidates(args.input)

    print(f"Loaded {len(candidates):,} candidates.")

    if not candidates:
        raise ValueError("No candidates were found.")

    # -----------------------------------------------------------------------
    # Create encoder
    # -----------------------------------------------------------------------

    encoder_kwargs = {}

    if args.device is not None:
        encoder_kwargs["device"] = args.device

    if args.batch_size is not None:
        encoder_kwargs["batch_size"] = args.batch_size

    print(f"Loading encoder: {args.encoder}")

    encoder = create_encoder(
        args.encoder,
        **encoder_kwargs,
    )

    # -----------------------------------------------------------------------
    # Encode
    # -----------------------------------------------------------------------

    print("Encoding candidate metadata...")

    embeddings = encoder.encode(texts)

    print(f"Embedding shape: {embeddings.shape}")

    # -----------------------------------------------------------------------
    # Save embeddings
    # -----------------------------------------------------------------------

    embeddings_path = output_dir / "embeddings.npy"

    np.save(
        embeddings_path,
        embeddings,
    )

    print(f"Saved embeddings to: {embeddings_path}")

    # -----------------------------------------------------------------------
    # Save candidate information
    # -----------------------------------------------------------------------

    candidates_path = output_dir / "candidates.json"

    with candidates_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            candidates,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved candidate metadata to: {candidates_path}")

    # -----------------------------------------------------------------------
    # Save configuration
    # -----------------------------------------------------------------------

    config = {
        "version": "v1",
        "encoder": encoder.name,
        "input_file": str(args.input),
        "num_candidates": len(candidates),
        "embedding_shape": list(embeddings.shape),
        "representation": {
            "fields": [
                "id",
                "docType",
                "docTitle",
                "autoCompletions.text",
            ],
            "autoCompletions": (
                "Only the text field of each autoCompletion is encoded; "
                "the completion type is discarded."
            ),
            "format": "fields concatenated with ' | '",
        },
    }

    config_path = output_dir / "config.json"

    with config_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            config,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved configuration to: {config_path}")

    print()
    print("Done.")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()