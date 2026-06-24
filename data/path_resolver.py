"""Path resolution for datalist entries."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Union

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


def find_datalist_file(datalist_dir: PathLike, split: str) -> str:
    """Find datalist file under datalist/<split>/ directory."""
    base = Path(datalist_dir) / split
    if base.is_file():
        return str(base)
    if base.is_dir():
        candidates = sorted(base.glob("*.txt"))
        if len(candidates) == 1:
            return str(candidates[0])
        if candidates:
            return str(candidates[0])
    raise FileNotFoundError(f"No datalist found for split '{split}' under {datalist_dir}")
