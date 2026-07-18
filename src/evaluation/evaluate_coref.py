#!/usr/bin/env python
"""
LeREaD Coreference Evaluation Script

Flexible evaluation of a single document or a batch folder, mirroring
evaluate.py (the extraction evaluator). Sibling script, not a --task flag on
evaluate.py: coref inputs (JSON clusters / docid-annotated HTML) and metrics
(MUC/B3/CEAFe/LEA) are different enough from the extraction evaluator's
(span HTML, span-F1) that a shared CLI would just be two disjoint arg sets
glued together.

System output can be either a `*_coref_clusters.json` file or a
`*_coref.html` file (docid attributes are read straight off the tags) —
both are produced by main_coref.py. Gold clusters are always derived live
from the annotated split (DATA_DIR/annotated/<split>/<filename>.html), by
reading each mention's ground-truth `docid` attribute — no separate gold
cluster file needs to be maintained.

Usage:
    # Single file evaluation (requires split parameter)
    python evaluate_coref.py single --split test \\
        --system-file output_coref/.../1989CanLII1415ONCA_coref_clusters.json

    # Batch folder evaluation (auto-detects split from folder name)
    python evaluate_coref.py batch \\
        --batch-folder output_coref/test_gpt-5.2_coref_fs6_random_gold/

    # Batch folder evaluation with explicit split
    python evaluate_coref.py batch \\
        --batch-folder output_coref/test_gpt-5.2_coref_fs6_random_gold/ --split test
"""

import argparse
import sys
import json
import io
import contextlib
import importlib.util
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from datetime import datetime

from bs4 import BeautifulSoup

from configs.config import SPLITS

# Add project root to path for imports (same depth convention as evaluate.py)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import evaluation utilities directly to avoid full src package import
spec = importlib.util.spec_from_file_location(
    "evaluation_coref_util",
    project_root / "src" / "evaluation" / "evaluation_coref.py"
)
evaluation_coref_util = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation_coref_util)
evaluate_coref_raw = evaluation_coref_util.evaluate_coref_raw
aggregate_raw = evaluation_coref_util.aggregate_raw
finalize_raw = evaluation_coref_util.finalize_raw
print_evaluation_table = evaluation_coref_util.print_evaluation_table

# Import config constants
spec_config = importlib.util.spec_from_file_location(
    "config",
    project_root / "configs" / "config.py"
)
config_module = importlib.util.module_from_spec(spec_config)
spec_config.loader.exec_module(config_module)
DATA_DIR = config_module.DATA_DIR


# =============================================================================
# Config
# =============================================================================

@dataclass
class CorefEvaluationConfig:
    mode: str  # "single" | "batch"
    split: Optional[str] = None
    system_file: Optional[str] = None
    batch_folder: Optional[str] = None
    data_dir: Path = Path("./data")
    verbose_per_file: bool = False

    def get_ground_truth_dir(self) -> Path:
        return Path(self.data_dir) / "annotated" / self.split

    def validate(self) -> None:
        if self.mode == "single":
            if not self.split:
                raise ValueError("--split is required for single file evaluation")
            if not self.system_file:
                raise ValueError("--system-file is required for single file evaluation")
        elif self.mode == "batch":
            if not self.batch_folder:
                raise ValueError("--batch-folder is required for batch evaluation")
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


# =============================================================================
# HTML / cluster helpers
# =============================================================================

def extract_mention_docid_mapping(html_content: str, parser: str = "html.parser") -> Dict[str, str]:
    """id -> docid for every tag in the document carrying both attributes."""
    soup = BeautifulSoup(html_content, parser)
    return {tag["id"]: tag["docid"] for tag in soup.find_all(attrs={"docid": True, "id": True})}


def dict_to_clusters(mapping: Dict[str, str]) -> List[List[str]]:
    clusters: Dict[str, List[str]] = {}
    for mention_id, docid in mapping.items():
        clusters.setdefault(docid, []).append(mention_id)
    return list(clusters.values())


def load_clusters(path: Path) -> List[List[str]]:
    """Load clusters from a `*_coref_clusters.json` file, or derive them from
    a docid-annotated HTML file (gold annotated file, or a `*_coref.html`
    system reconstruction)."""
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    if path.suffix == ".html":
        with open(path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return dict_to_clusters(extract_mention_docid_mapping(html_content))
    raise ValueError(f"Unsupported file type for clusters: {path}")


def derive_doc_stem(system_path: Path) -> str:
    """Strip known coref-output suffixes to recover the original filename
    stem, used to look up the matching gold file."""
    name = system_path.stem  # already strips .json / .html
    for suffix in ("_coref_clusters", "_coref_mapping", "_coref"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def extract_split_from_folder_name(folder_name: str) -> Optional[str]:
    """Expected patterns, matching main_coref.py's run_tag(), e.g.
    'test_gpt-5.2_coref_fs6_random_gold' or
    'dev_gpt-5.2_coref_fs6_greedy_from-<extraction_folder>-final'."""
    parts = folder_name.split("_")
    if parts and parts[0] in ["train", "test", "dev"]:
        return parts[0]
    return None


# =============================================================================
# Logging helpers
# =============================================================================

def save_evaluation_log(folder_path: Path, log_content: str) -> None:
    log_file = folder_path / "evaluation_coref_results.txt"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(log_content)
    print(f"✓ Saved evaluation results to: {log_file}")


def save_version_info(folder_path: Path, config: CorefEvaluationConfig, mode: str, details: str = "") -> None:
    version_file = folder_path / "version_coref.txt"
    content = (
        "Coref Evaluation Version Information\n"
        "================================\n"
        f"Timestamp: {datetime.now().isoformat()}\n"
        f"Mode: {mode}\n"
        f"Split: {config.split}\n\n"
    )
    if details:
        content += f"Details:\n{details}\n"
    with open(version_file, "w") as f:
        f.write(content)
    print(f"✓ Saved evaluation info to: {version_file}")


# =============================================================================
# Single-file evaluation
# =============================================================================

def run_single_file_evaluation(config: CorefEvaluationConfig):
    system_path = Path(config.system_file)
    if not system_path.exists():
        raise FileNotFoundError(f"System file not found: {system_path}")

    doc_stem = derive_doc_stem(system_path)
    gold_dir = config.get_ground_truth_dir()
    gold_path = gold_dir / f"{doc_stem}.html"
    if not gold_path.exists():
        raise FileNotFoundError(f"Gold file not found: {gold_path}")

    header = (
        f"\n{'='*70}\n"
        f"SINGLE FILE COREF EVALUATION\n"
        f"{'='*70}\n"
        f"Split: {config.split}\n"
        f"Gold:   {gold_path}\n"
        f"System: {system_path}\n"
        f"{'='*70}\n"
    )
    print(header)

    gold_clusters = load_clusters(gold_path)
    system_clusters = load_clusters(system_path)

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        raw = evaluate_coref_raw(gold_clusters, system_clusters)
        scores = finalize_raw(raw)
        print_evaluation_table(scores, title=f"Coreference Evaluation — {doc_stem}")
    captured = buffer.getvalue()
    print(captured)

    full_log = header + captured
    save_evaluation_log(system_path.parent, full_log)


# =============================================================================
# Batch evaluation
# =============================================================================

def collect_cluster_pairs_from_batch_folder(
    batch_folder: Path,
    data_dir: Path,
    split: str,
    verbose: bool = False,
) -> List[Tuple[Path, Path]]:
    """Collect (gold_path, system_path) pairs from a batch folder.

    Handles two structures (same convention as evaluate.py):
    1. Direct files: `*_coref_clusters.json` (preferred) or `*_coref.html`
       directly inside batch_folder
    2. Subfolder-per-file structure, as produced by main_coref.py's split
       mode: batch_folder/<filename>/<filename>_coref_clusters.json
    """
    gold_dir = data_dir / "annotated" / split
    if not gold_dir.exists():
        raise ValueError(f"Gold directory does not exist: {gold_dir}")
    gold_files = {p.stem: p for p in gold_dir.glob("*.html")}

    if verbose:
        print(f"Found {len(gold_files)} gold files in {gold_dir}")

    def pair_up(system_files: List[Path]) -> List[Tuple[Path, Path]]:
        pairs = []
        for system_file in system_files:
            stem = derive_doc_stem(system_file)
            if stem in gold_files:
                pairs.append((gold_files[stem], system_file))
                if verbose:
                    print(f"  Paired: {stem}")
            else:
                print(f"  ⚠ No matching gold file for: {stem}")
        return pairs

    # Strategy 1: direct files
    direct_files = list(batch_folder.glob("*_coref_clusters.json"))
    if not direct_files:
        direct_files = list(batch_folder.glob("*_coref.html"))
    direct_files = [f for f in direct_files if f.is_file()]
    if direct_files:
        if verbose:
            print(f"Found {len(direct_files)} direct coref output files")
        return pair_up(direct_files)

    # Strategy 2: subfolder-per-file
    subfolders = [d for d in batch_folder.iterdir() if d.is_dir()]
    if subfolders:
        if verbose:
            print(f"Found {len(subfolders)} subfolders")
        system_files = []
        for subfolder in subfolders:
            candidates = list(subfolder.glob("*_coref_clusters.json")) or list(subfolder.glob("*_coref.html"))
            if candidates:
                system_files.append(candidates[0])
            elif verbose:
                print(f"  ⚠ No coref output found in: {subfolder.name}")
        return pair_up(system_files)

    raise ValueError(f"No coref cluster/HTML files or subfolders found in {batch_folder}")


def run_batch_evaluation(config: CorefEvaluationConfig):
    batch_folder = Path(config.batch_folder)
    if not batch_folder.exists():
        raise FileNotFoundError(f"Batch folder not found: {batch_folder}")

    split = config.split
    if not split:
        detected_split = extract_split_from_folder_name(batch_folder.name)
        if detected_split:
            split = detected_split
            print(f"Auto-detected split from folder name: {split}")
        else:
            raise ValueError(
                f"Could not auto-detect split from folder name: {batch_folder.name}\n"
                f"Please provide split parameter explicitly."
            )

    header = (
        f"\n{'='*70}\n"
        f"BATCH COREF EVALUATION\n"
        f"{'='*70}\n"
        f"Batch folder: {batch_folder}\n"
        f"Split: {split}\n"
        f"{'='*70}\n"
    )
    print(header)

    pairs = collect_cluster_pairs_from_batch_folder(batch_folder, config.data_dir, split, verbose=False)
    if not pairs:
        raise ValueError(f"No file pairs found in batch folder: {batch_folder}")
    print(f"Found {len(pairs)} evaluation pair(s)\n")

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        per_file_raws = []
        for gold_path, system_path in pairs:
            gold_clusters = load_clusters(gold_path)
            system_clusters = load_clusters(system_path)
            raw = evaluate_coref_raw(gold_clusters, system_clusters)
            per_file_raws.append(raw)
            if config.verbose_per_file:
                print_evaluation_table(finalize_raw(raw), title=f"{gold_path.stem}")

        # Corpus-level aggregation: sum raw counts, then compute one overall
        # P/R/F per metric (standard CoNLL-scorer convention), rather than
        # averaging per-document F1s.
        aggregated_scores = finalize_raw(aggregate_raw(per_file_raws))
        print_evaluation_table(aggregated_scores, title=f"Aggregate over {len(pairs)} document(s)")
    captured = buffer.getvalue()

    print(captured)

    full_log = header + f"Found {len(pairs)} evaluation pair(s)\n\n" + captured
    save_evaluation_log(batch_folder, full_log)

    return aggregated_scores


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="LeREaD Coreference Evaluation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Evaluation mode")

    # ========== SINGLE FILE EVALUATION ==========
    single_parser = subparsers.add_parser("single", help="Evaluate a single system coref output")
    single_parser.add_argument("--split", required=True, choices=SPLITS,
                                help="Ground truth split (required for single file evaluation)")
    single_parser.add_argument("--system-file", required=True,
                                help="Path to a *_coref_clusters.json or *_coref.html system output file")
    single_parser.add_argument("--data-dir", default="./data", help="Root data directory (default: ./data)")
    single_parser.add_argument("--verbose", action="store_true", help="Verbose output")

    # ========== BATCH EVALUATION ==========
    batch_parser = subparsers.add_parser("batch", help="Evaluate a batch folder with multiple documents")
    batch_parser.add_argument("--batch-folder", required=True, help="Path to a main_coref.py output folder")
    batch_parser.add_argument("--split", default=None, choices=SPLITS,
                               help="Ground truth split (optional, auto-detected from folder name if not provided)")
    batch_parser.add_argument("--data-dir", default="./data", help="Root data directory (default: ./data)")
    batch_parser.add_argument("--verbose", action="store_true",
                               help="Also print a per-document table before the aggregate")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "single":
        config = CorefEvaluationConfig(
            mode="single",
            split=args.split,
            system_file=args.system_file,
            data_dir=Path(args.data_dir),
            verbose_per_file=args.verbose,
        )
        config.validate()
        run_single_file_evaluation(config)

    elif args.command == "batch":
        config = CorefEvaluationConfig(
            mode="batch",
            split=args.split,
            batch_folder=args.batch_folder,
            data_dir=Path(args.data_dir),
            verbose_per_file=args.verbose,
        )
        config.validate()
        run_batch_evaluation(config)


if __name__ == "__main__":
    main()