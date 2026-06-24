"""Build validation cache: noisy/clean point clouds from meshes."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.mesh_utils import sample_points_from_mesh_path
from data.path_resolver import find_datalist_file, read_datalist, resolve_mesh_path, resolve_val_clean_path, resolve_val_noisy_path
from data.transform import add_laplace_noise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="./dataset_clean")
    parser.add_argument("--datalist_dir", default="./datalist")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--val_cache_root", default="./val_cache")
    parser.add_argument("--num_points", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20260613)
    args = parser.parse_args()

    np.random.seed(args.seed)
    list_file = find_datalist_file(args.datalist_dir, args.split)
    lines = read_datalist(list_file)

    for line in tqdm(lines, desc="build_val_cache"):
        mesh_path = resolve_mesh_path(args.dataset_root, line)
        clean, _ = sample_points_from_mesh_path(
            mesh_path,
            num_samples=args.num_points,
            num_vertex_samples=0,
            normalize=True,
        )
        noise_std = float(np.random.uniform(0.005, 0.020))
        noisy = add_laplace_noise(clean, noise_std)

        clean_path = resolve_val_clean_path(args.val_cache_root, line)
        noisy_path = resolve_val_noisy_path(args.val_cache_root, line)
        os.makedirs(os.path.dirname(clean_path), exist_ok=True)
        os.makedirs(os.path.dirname(noisy_path), exist_ok=True)
        np.save(clean_path, clean.astype(np.float32))
        np.save(noisy_path, noisy.astype(np.float32))

    print(f"Built {len(lines)} validation pairs under {args.val_cache_root}")


if __name__ == "__main__":
    main()
