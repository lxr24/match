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
    """Find datalist file for a split.

    Supported layouts:
      - datalist/train.txt              (flat)
      - datalist/train/                 (file named after split)
      - datalist/train/<any>.txt        (directory with txt files)
    """
    root = Path(datalist_dir)

    # Flat: datalist/train.txt
    flat = root / f"{split}.txt"
    if flat.is_file():
        return str(flat)

    # datalist/train as a file (unusual but supported)
    direct = root / split
    if direct.is_file():
        return str(direct)

    # datalist/train/*.txt
    if direct.is_dir():
        candidates = sorted(direct.glob("*.txt"))
        if candidates:
            return str(candidates[0])

    raise FileNotFoundError(
        f"No datalist found for split '{split}' under {datalist_dir}. "
        f"Expected one of: {flat}, {direct}, or {direct}/*.txt"
    )
