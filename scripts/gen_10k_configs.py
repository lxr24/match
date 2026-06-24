#!/usr/bin/env python3
"""Generate 10k training configs from 5k templates."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
pairs = [
    ("train_vm.yaml", "train_vm_10k.yaml"),
    ("train_cvm.yaml", "train_cvm_10k.yaml"),
    ("train_core.yaml", "train_core_10k.yaml"),
    ("train_joint.yaml", "train_joint_10k.yaml"),
]

for src_name, dst_name in pairs:
    src = (ROOT / "configs" / src_name).read_text(encoding="utf-8")
    dst = src.replace("train_strat5k_seed123", "train_strat10k_seed123")
    dst = dst.replace("num_files: 5000", "num_files: 10000")
    dst = dst.replace("short5k", "short10k")
    (ROOT / "configs" / dst_name).write_text(dst, encoding="utf-8")
    print(f"Wrote configs/{dst_name}")
