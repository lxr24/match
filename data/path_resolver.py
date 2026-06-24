"""Path resolution for datalist entries."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Union

from data.paths import DATALIST_SPLIT_ALIASES

PathLike = Union[str, Path]


def read_datalist(path: PathLike) -> List[str]:
    lines: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line and not line.startswith("#"):
                lines.append(line)
    return lines


def resolve_mesh_path(dataset_root: PathLike, datalist_line: str) -> str:
    return os.path.join(str(dataset_root), datalist_line, "models", "model_normalized.obj")


def resolve_noisy_path(test_root: PathLike, datalist_line: str) -> str:
    return os.path.join(str(test_root), datalist_line, "noisy.npy")


def resolve_output_path(output_root: PathLike, datalist_line: str) -> str:
    return os.path.join(str(output_root), datalist_line, "denoised.npy")


def resolve_val_noisy_path(val_cache_root: PathLike, datalist_line: str) -> str:
    return os.path.join(str(val_cache_root), "noisy", datalist_line, "noisy.npy")


def resolve_val_clean_path(val_cache_root: PathLike, datalist_line: str) -> str:
    return os.path.join(str(val_cache_root), "clean", datalist_line, "clean.npy")


def _candidate_paths(root: Path, stem: str) -> List[Path]:
    return [
        root / f"{stem}.txt",
        root / stem,
        root / stem / f"{stem}.txt",
    ]


def find_datalist_file(datalist_dir: PathLike, split: str) -> str:
    """Find datalist file for a split.

    Supported layouts (examples for split=train):
      - datalist/train.txt
      - datalist/validate.txt   (alias for validation)
      - datalist/train/
      - datalist/train/*.txt
    """
    root = Path(datalist_dir)
    stems = DATALIST_SPLIT_ALIASES.get(split, [split])
    tried: List[str] = []

    for stem in stems:
        for path in _candidate_paths(root, stem):
            tried.append(str(path))
            if path.is_file():
                return str(path)
        # datalist/<stem>/*.txt directory
        stem_dir = root / stem
        if stem_dir.is_dir():
            candidates = sorted(stem_dir.glob("*.txt"))
            if candidates:
                return str(candidates[0])
            tried.append(str(stem_dir / "*.txt"))

    raise FileNotFoundError(
        f"No datalist found for split '{split}' under {datalist_dir}. "
        f"Tried: {', '.join(tried)}"
    )
