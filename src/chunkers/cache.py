"""Pure I/O"""
import json
from pathlib import Path
from configs.config import CHUNK_CACHE_DIR


def _cache_path(method: str, split: str, filename: str) -> Path:
    return CHUNK_CACHE_DIR / method / split / f"{filename}.json"


def cache_exists(method: str, split: str, filename: str) -> bool:
    return _cache_path(method, split, filename).is_file()


def load_cache(method: str, split: str, filename: str) -> list[list[str]]:
    with _cache_path(method, split, filename).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_cache(method: str, split: str, filename: str, token_chunks: list[list[str]]) -> None:
    path = _cache_path(method, split, filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(token_chunks, f)