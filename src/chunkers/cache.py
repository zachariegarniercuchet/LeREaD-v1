"""Pure I/O"""
import json
from pathlib import Path


def _cache_path(cache_dir: str, method: str, split: str, filename: str) -> Path:
    return Path(cache_dir) / method / split / f"{filename}.json"


def cache_exists(method: str, cache_dir: str, split: str, filename: str) -> bool:
    return _cache_path(cache_dir, method, split, filename).is_file()


def load_cache(method: str, cache_dir: str, split: str, filename: str) -> list[list[str]]:
    with _cache_path(cache_dir, method, split, filename).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_cache(method: str, cache_dir: str, split: str, filename: str, token_chunks: list[list[str]]) -> None:
    path = _cache_path(cache_dir, method, split, filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(token_chunks, f)