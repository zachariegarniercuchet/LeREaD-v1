"""Main processing script for reference *resolution* (stage 3): candidate URI
retrieval via embedding similarity search over the pre-computed candidate pool.

Mirrors main_extraction.py / main_coref.py in structure and CLI so the three
pipeline stages stay easy to read side by side.

Pipeline recap:
  stage 1 (main_extraction.py): raw text  -> extracted <manual_label>/<auto_label>
                                 mentions.
  stage 2 (main_coref.py):      mentions  -> docid assigned per mention via an
                                 assistant + a growing ReferenceProfileRegistry;
                                 reconstructed as "<docname>_coref.html".
  stage 3 (this file):          "<docname>_coref.html" -> re-parsed to rebuild
                                 the same ReferenceProfileRegistry (no LLM calls,
                                 purely deterministic), then every profile is
                                 embedded and matched against the pre-computed
                                 candidate pool (see precompute_embeddings_metadata.py)
                                 to assign a `uri` attribute to every resolved
                                 mention tag; reconstructed as
                                 "<docname>_resolution.html".

Input layout expected under --dir/--folder:

    output_coref/
        <folder>/
            <docname_1>/
                <docname_1>_coref.html
            <docname_2>/
                <docname_2>_coref.html
            ...

Output written alongside the input, per document:

    <docname>_resolution.html            (same HTML, `uri` added to every
                                           resolved manual_label/auto_label tag)
    <docname>_resolution_mapping.json    (docid -> {uri, score, ...}, for
                                           inspection/debugging)

GOLD MODE (--gold): instead of reading a stage-2 run folder, the gold annotated
files of a split are used as input. Every `uri` attribute is stripped from them
(so the resolver cannot see the answer), then the exact same resolution function
is applied. Output goes to a fresh folder under --dir:

    output_coref/
        <split>_gold_resolution/
            <docname_1>/
                <docname_1>_coref.html          (uri-stripped gold = the input)
                <docname_1>_resolution.html
                <docname_1>_resolution_mapping.json
            ...

which is directly evaluable with evaluate_resolution.py:

    python src/evaluation/evaluate_resolution.py batch \
        --batch-folder output_coref/test_gold_resolution --split test


python -m src.main_resolution --folder test_gemma4_coref_fs6_random_gold --dir output_coref --encoder bge-s
python -m src.main_resolution --gold --split test --encoder bge-s
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from bs4 import BeautifulSoup
from tqdm import tqdm

from src.ann_extractor import extract_parent_level_annotations
from src.encoder_models import create_encoder
from src.htmlLabel import ReferenceMention
from src.rpr import ReferenceProfile, ReferenceProfileRegistry, normalize_docid


# =============================================================================
# Candidate pool loading
# =============================================================================

def load_candidate_pool(cache_dir: Path) -> Tuple[np.ndarray, List[dict]]:
    """Load the embeddings + metadata produced by precompute_embeddings_metadata.py."""

    embeddings_path = cache_dir / "embeddings.npy"
    candidates_path = cache_dir / "candidates.json"

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Missing embeddings file: {embeddings_path}")
    if not candidates_path.exists():
        raise FileNotFoundError(f"Missing candidates file: {candidates_path}")

    embeddings = np.load(embeddings_path)

    with candidates_path.open("r", encoding="utf-8") as f:
        candidates = json.load(f)

    if len(candidates) != embeddings.shape[0]:
        raise ValueError(
            f"candidates.json has {len(candidates)} entries but embeddings.npy "
            f"has {embeddings.shape[0]} rows - the cache appears inconsistent."
        )

    return embeddings, candidates


# =============================================================================
# Profile -> query text
# =============================================================================

def build_profile_text(profile: ReferenceProfile, include_fragments: bool = False) -> str:
    """
    Build the textual representation of a ReferenceProfile used to query the
    candidate pool. Mirrors build_metadata_text() in
    precompute_embeddings_metadata.py (docType | docTitle | reference texts),
    but is built from the profile accumulated across every mention sharing a
    docid instead of from raw CanLII metadata.
    """

    parts: List[str] = []

    if profile.doc_type:
        parts.append(str(profile.doc_type))

    if profile.main_title:
        parts.append(str(profile.main_title))

    for title in profile.alternative_titles:
        if title != profile.main_title:
            parts.append(str(title))

    for citation in profile.citations:
        parts.append(str(citation))

    if include_fragments:
        for fragment in profile.fragments_mentioned:
            parts.append(str(fragment))

    for author in profile.authors:
        parts.append(str(author))

    return " | ".join(parts)


# =============================================================================
# Similarity search
# =============================================================================

def _normalize_doctype(value: Optional[str]) -> str:
    if not value:
        return ""
    return str(value).strip().upper().replace(" ", "_")


def _select_candidate_pool(
    profile: ReferenceProfile,
    embeddings: np.ndarray,
    candidates: List[dict],
    filter_by_doctype: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (candidate_indices, candidate_embeddings) to search over for this
    profile. Falls back to the full pool whenever filtering isn't requested,
    the profile has no doc_type, or no candidate matches the doc_type (rather
    than silently failing to resolve).
    """

    if not filter_by_doctype or not profile.doc_type:
        idx = np.arange(len(candidates))
        return idx, embeddings

    target = _normalize_doctype(profile.doc_type)
    matching = [
        i for i, c in enumerate(candidates)
        if _normalize_doctype(c.get("docType")) == target
    ]

    if not matching:
        idx = np.arange(len(candidates))
        return idx, embeddings

    idx = np.array(matching)
    return idx, embeddings[idx]


def _top_k(query_vec: np.ndarray, matrix: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Cosine-similarity top-k search of query_vec against the rows of matrix."""

    query_vec = np.asarray(query_vec, dtype=float)
    matrix = np.asarray(matrix, dtype=float)

    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    matrix_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix_norm = matrix / (matrix_norms + 1e-12)

    scores = matrix_norm @ query_norm
    k = max(1, min(k, len(scores)))
    top_idx = np.argsort(-scores)[:k]

    return top_idx, scores[top_idx]


# =============================================================================
# HTML reconstruction
# =============================================================================

def apply_uris_to_html(html_content: str, mapping: Dict[str, dict], parser: str = "html.parser") -> str:
    """
    Reconstruct the document, adding a `uri` attribute to every
    manual_label/auto_label tag whose (normalized) docid was resolved.

    `uri` is inserted as the first attribute so the output matches the
    "<manual_label uri=... docid=... id=... ...>" ordering from the spec.
    Tags without a docid (e.g. nested title/fragment mentions, or mentions
    that failed coref resolution) are left untouched.
    """

    soup = BeautifulSoup(html_content, parser)

    for tag in soup.find_all(["manual_label", "auto_label"], attrs={"docid": True}):
        docid_norm = normalize_docid(tag.get("docid"))
        entry = mapping.get(docid_norm)
        if entry is None:
            continue

        attrs = dict(tag.attrs)
        attrs.pop("uri", None)
        tag.attrs = {"uri": entry["uri"], **attrs}

    return str(soup)


def strip_uris_from_html(html_content: str, parser: str = "html.parser") -> Tuple[str, int]:
    """
    Remove every `uri` attribute from manual_label/auto_label tags.

    Used in gold mode so the resolver never sees the ground-truth uri it is
    supposed to predict. Everything else (docid, labelname, parent, titletype,
    text, …) is preserved untouched.

    Returns:
        (stripped_html, n_removed)
    """

    soup = BeautifulSoup(html_content, parser)

    removed = 0
    for tag in soup.find_all(["manual_label", "auto_label"], attrs={"uri": True}):
        del tag.attrs["uri"]
        removed += 1

    return str(soup), removed


# =============================================================================
# Core processing
# =============================================================================

def process_document_resolution(
    html_content: str,
    encoder,
    embeddings: np.ndarray,
    candidates: List[dict],
    filter_by_doctype: bool = True,
    include_fragments: bool = False,
    score_threshold: Optional[float] = None,
    debug_topk: int = 3,
) -> Tuple[str, Dict[str, dict], int]:
    """
    Rebuild the ReferenceProfileRegistry for one coref-resolved document
    (deterministic, no assistant calls), retrieve a matching candidate uri for
    every resolved profile, and return the HTML with `uri` attributes injected.

    Returns:
        output_html: the reconstructed HTML with `uri` attributes added.
        mapping: normalized docid -> {"uri", "score", "candidate_id",
                 "candidate_title", "query_text", "topk"}.
        unresolved: number of docid-bearing profiles that could not be
                    confidently assigned a uri (empty query text, empty
                    candidate pool, or below --score-threshold).
    """

    mentions = extract_parent_level_annotations(html_content)

    rpr = ReferenceProfileRegistry()
    for mention in mentions:
        rpr.update_from_mention(ReferenceMention(mention.html_str))

    unresolved = 0

    # Only profiles that actually got a docid during the coref stage are
    # worth resolving; the rest were never coref-resolved and their tags
    # have no `docid` attribute for apply_uris_to_html() to match against.
    resolvable_profiles: List[ReferenceProfile] = []
    texts: List[str] = []

    for profile in rpr:
        if not profile.docid:
            continue

        text = build_profile_text(profile, include_fragments=include_fragments)
        if not text.strip():
            unresolved += 1
            continue

        resolvable_profiles.append(profile)
        texts.append(text)

    mapping: Dict[str, dict] = {}

    if resolvable_profiles:
        profile_embeddings = encoder.encode(texts)

        for profile, profile_embedding, text in zip(resolvable_profiles, profile_embeddings, texts):
            candidate_idx, candidate_pool = _select_candidate_pool(
                profile, embeddings, candidates, filter_by_doctype
            )

            if len(candidate_idx) == 0:
                unresolved += 1
                continue

            top_local_idx, top_scores = _top_k(profile_embedding, candidate_pool, k=debug_topk)
            best_score = float(top_scores[0])

            if score_threshold is not None and best_score < score_threshold:
                unresolved += 1
                continue

            best_candidate = candidates[candidate_idx[top_local_idx[0]]]

            mapping[normalize_docid(profile.docid)] = {
                "uri": best_candidate["original_url"],
                "score": best_score,
                "candidate_id": best_candidate.get("id"),
                "candidate_title": best_candidate.get("docTitle"),
                "query_text": text,
                "topk": [
                    {
                        "uri": candidates[candidate_idx[i]]["original_url"],
                        "score": float(s),
                        "candidate_title": candidates[candidate_idx[i]].get("docTitle"),
                    }
                    for i, s in zip(top_local_idx, top_scores)
                ],
            }

    output_html = apply_uris_to_html(html_content, mapping)

    return output_html, mapping, unresolved


def process_folder_resolution(
    folder: str,
    base_dir: Path,
    encoder,
    embeddings: np.ndarray,
    candidates: List[dict],
    filter_by_doctype: bool,
    include_fragments: bool,
    score_threshold: Optional[float],
    debug_topk: int,
    overwrite: bool,
) -> None:

    folder_dir = base_dir / folder
    if not folder_dir.exists():
        raise FileNotFoundError(f"Folder not found: {folder_dir}")

    doc_dirs = sorted(d for d in folder_dir.iterdir() if d.is_dir())
    print(f"[Setup] Found {len(doc_dirs)} document folder(s) in {folder_dir}\n")

    total_resolved = 0
    total_unresolved = 0

    for doc_dir in tqdm(doc_dirs, desc="Resolving documents"):
        coref_files = sorted(doc_dir.glob("*_coref.html"))
        if not coref_files:
            print(f"  ⚠️  No *_coref.html found in {doc_dir.name} — skipping")
            continue

        coref_path = coref_files[0]
        doc_name = coref_path.name[: -len("_coref.html")]

        output_path = doc_dir / f"{doc_name}_resolution.html"
        mapping_path = doc_dir / f"{doc_name}_resolution_mapping.json"

        if output_path.exists() and not overwrite:
            print(f"  ⏭  {doc_name}: resolution output already exists — skipping")
            continue

        print(f"\n{'='*80}")
        print(f"Resolving: {doc_name}")
        print(f"{'='*80}")

        with coref_path.open("r", encoding="utf-8") as f:
            html_content = f.read()

        output_html, mapping, unresolved = process_document_resolution(
            html_content=html_content,
            encoder=encoder,
            embeddings=embeddings,
            candidates=candidates,
            filter_by_doctype=filter_by_doctype,
            include_fragments=include_fragments,
            score_threshold=score_threshold,
            debug_topk=debug_topk,
        )

        with output_path.open("w", encoding="utf-8") as f:
            f.write(output_html)

        with mapping_path.open("w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)

        print(f"  ✓ Resolved {len(mapping)} profile(s), {unresolved} unresolved")
        print(f"  ✓ Saved {output_path.name} and {mapping_path.name}")

        total_resolved += len(mapping)
        total_unresolved += unresolved

    print(f"\n{'='*80}")
    print("✓ RESOLUTION COMPLETE")
    print(f"✓ Folder: {folder_dir}")
    print(f"✓ Total profiles resolved: {total_resolved}")
    print(f"✓ Total profiles unresolved: {total_unresolved}")
    print("="*80)


# =============================================================================
# Gold mode
# =============================================================================

def process_gold_split_resolution(
    split: str,
    gold_dir: Path,
    out_dir: Path,
    encoder,
    embeddings: np.ndarray,
    candidates: List[dict],
    filter_by_doctype: bool,
    include_fragments: bool,
    score_threshold: Optional[float],
    debug_topk: int,
    overwrite: bool,
) -> None:
    """
    Oracle run: take every gold annotated file of a split, strip its `uri`
    attributes, and run the exact same resolution as on stage-2 output.

    This isolates stage 3: the docids come from the gold annotation instead of
    from the coref model, so the resulting uri accuracy is an upper bound on
    what the full pipeline can achieve.

    Layout written under out_dir (one folder per document, same shape as a
    stage-2 run folder so the evaluation script works unchanged):

        <out_dir>/<docname>/<docname>_coref.html            (stripped input)
        <out_dir>/<docname>/<docname>_resolution.html
        <out_dir>/<docname>/<docname>_resolution_mapping.json
    """

    split_dir = gold_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Gold split directory not found: {split_dir}")

    gold_files = sorted(split_dir.glob("*.html"))
    if not gold_files:
        raise FileNotFoundError(f"No .html files found in {split_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Setup] Gold split   : {split_dir}")
    print(f"[Setup] Found {len(gold_files)} gold document(s)")
    print(f"[Setup] Output folder: {out_dir}\n")

    total_resolved = 0
    total_unresolved = 0
    total_stripped = 0

    for gold_path in tqdm(gold_files, desc="Resolving gold documents"):
        doc_name = gold_path.stem

        doc_dir = out_dir / doc_name
        input_path = doc_dir / f"{doc_name}_coref.html"
        output_path = doc_dir / f"{doc_name}_resolution.html"
        mapping_path = doc_dir / f"{doc_name}_resolution_mapping.json"

        if output_path.exists() and not overwrite:
            print(f"  ⏭  {doc_name}: resolution output already exists — skipping")
            continue

        print(f"\n{'='*80}")
        print(f"Resolving (gold): {doc_name}")
        print(f"{'='*80}")

        with gold_path.open("r", encoding="utf-8") as f:
            gold_html = f.read()

        stripped_html, n_removed = strip_uris_from_html(gold_html)
        print(f"  • Stripped {n_removed} gold uri attribute(s) from the input")

        output_html, mapping, unresolved = process_document_resolution(
            html_content=stripped_html,
            encoder=encoder,
            embeddings=embeddings,
            candidates=candidates,
            filter_by_doctype=filter_by_doctype,
            include_fragments=include_fragments,
            score_threshold=score_threshold,
            debug_topk=debug_topk,
        )

        doc_dir.mkdir(parents=True, exist_ok=True)

        with input_path.open("w", encoding="utf-8") as f:
            f.write(stripped_html)

        with output_path.open("w", encoding="utf-8") as f:
            f.write(output_html)

        with mapping_path.open("w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)

        print(f"  ✓ Resolved {len(mapping)} profile(s), {unresolved} unresolved")
        print(f"  ✓ Saved {output_path.name} and {mapping_path.name}")

        total_resolved += len(mapping)
        total_unresolved += unresolved
        total_stripped += n_removed

    print(f"\n{'='*80}")
    print("✓ GOLD RESOLUTION COMPLETE")
    print(f"✓ Split: {split}")
    print(f"✓ Output folder: {out_dir}")
    print(f"✓ Gold uri attributes hidden: {total_stripped}")
    print(f"✓ Total profiles resolved: {total_resolved}")
    print(f"✓ Total profiles unresolved: {total_unresolved}")
    print("="*80)


# =============================================================================
# Arg parser
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 3: resolve every reference profile in a coref-resolved "
            "document to a candidate uri via embedding similarity search "
            "over the pre-computed candidate pool."
        )
    )

    parser.add_argument(
        "--folder",
        default=None,
        help=(
            "Name of the stage-2 run folder under --dir, e.g. "
            "test_gemma4_coref_fs6_random_gold. Mutually exclusive with --gold."
        ),
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("output_coref"),
        help="Root directory containing --folder (and where --gold output is written).",
    )

    # --- gold (oracle) mode ---------------------------------------------------
    parser.add_argument(
        "--gold",
        action="store_true",
        help=(
            "Run stage 3 on the gold annotated files of --split instead of on a "
            "stage-2 run folder. Gold `uri` attributes are stripped before "
            "resolution."
        ),
    )
    parser.add_argument(
        "--split",
        choices=["train", "test", "dev", "incoming"],
        default=None,
        help="Gold split to use with --gold.",
    )
    parser.add_argument(
        "--gold-dir",
        type=Path,
        default=Path("data/annotated"),
        help="Root of the gold annotations (default: data/annotated).",
    )
    parser.add_argument(
        "--output-folder",
        default=None,
        help=(
            "Name of the output folder under --dir for --gold mode "
            "(default: <split>_gold_resolution)."
        ),
    )

    parser.add_argument("--encoder", choices=["bge-s", "splade"], default="bge-s")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=(
            "Candidate pool embeddings directory. Defaults to "
            "cache/candidate_pool_embeddings_<encoder>_v1."
        ),
    )
    parser.add_argument("--device", default=None, help="PyTorch device, e.g. cuda or cpu.")
    parser.add_argument("--batch-size", type=int, default=None)

    parser.add_argument(
        "--no-filter-by-doctype",
        action="store_true",
        help="Disable restricting candidates to the profile's doc_type.",
    )
    parser.add_argument(
        "--include-fragments",
        action="store_true",
        help="Include fragment mentions in the profile's query text.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="Minimum cosine similarity required to assign a uri. Default: no threshold.",
    )
    parser.add_argument(
        "--debug-topk",
        type=int,
        default=3,
        help="Number of top candidates recorded in the mapping json for inspection.",
    )

    parser.add_argument("--overwrite", action="store_true")

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    # --- input mode validation ------------------------------------------------
    if args.gold and args.folder:
        parser.error("--gold and --folder are mutually exclusive.")
    if not args.gold and not args.folder:
        parser.error("Provide either --folder (stage-2 run) or --gold --split.")
    if args.gold and not args.split:
        parser.error("--gold requires --split (train/test/dev/incoming).")

    cache_dir = args.cache_dir or Path(f"cache/candidate_pool_embeddings_{args.encoder}_v1")

    print(f"[Setup] Loading candidate pool from: {cache_dir}")
    embeddings, candidates = load_candidate_pool(cache_dir)
    print(f"  ✓ Loaded {len(candidates):,} candidates, embeddings shape {embeddings.shape}\n")

    encoder_kwargs: Dict[str, Any] = {}
    if args.device is not None:
        encoder_kwargs["device"] = args.device
    if args.batch_size is not None:
        encoder_kwargs["batch_size"] = args.batch_size

    print(f"[Setup] Loading encoder: {args.encoder}")
    encoder = create_encoder(args.encoder, **encoder_kwargs)
    print()

    if args.gold:
        output_folder = args.output_folder or f"{args.split}_gold_resolution"
        process_gold_split_resolution(
            split=args.split,
            gold_dir=args.gold_dir,
            out_dir=args.dir / output_folder,
            encoder=encoder,
            embeddings=embeddings,
            candidates=candidates,
            filter_by_doctype=not args.no_filter_by_doctype,
            include_fragments=args.include_fragments,
            score_threshold=args.score_threshold,
            debug_topk=args.debug_topk,
            overwrite=args.overwrite,
        )
    else:
        process_folder_resolution(
            folder=args.folder,
            base_dir=args.dir,
            encoder=encoder,
            embeddings=embeddings,
            candidates=candidates,
            filter_by_doctype=not args.no_filter_by_doctype,
            include_fragments=args.include_fragments,
            score_threshold=args.score_threshold,
            debug_topk=args.debug_topk,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()