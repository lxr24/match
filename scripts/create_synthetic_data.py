"""Generate minimal synthetic dataset for smoke tests."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import trimesh


def make_sphere_mesh(path: str, seed: int = 0):
    rng = np.random.default_rng(seed)
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    mesh.vertices += rng.normal(0, 0.002, size=mesh.vertices.shape)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mesh.export(path)


def write_datalists(out_dir: Path, synsets: list, counts: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    all_train = []
    all_val = []
    all_test = []

    for synset in synsets:
        n = counts.get(synset, 5)
        for i in range(n):
            mid = f"model_{synset}_{i:04d}"
            line = f"shapenet/{synset}/{mid}"
            all_train.append(line)

    rng = np.random.default_rng(123)
    rng.shuffle(all_train)
    all_val = all_train[: min(10, len(all_train))]
    all_test = all_train[10 : min(20, len(all_train))]
    train_lines = all_train[20:]

    for split, lines in (("train", train_lines), ("validation", all_val), ("test", all_test)):
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        with open(split_dir / f"{split}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    return train_lines, all_val, all_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--num_synsets", type=int, default=3)
    args = parser.parse_args()

    root = Path(args.root)
    synsets = [f"{i:08d}" for i in range(4300000, 4300000 + args.num_synsets)]

    all_lines = []
    for synset in synsets:
        for i in range(8):
            mid = f"model_{synset}_{i:04d}"
            line = f"shapenet/{synset}/{mid}"
            all_lines.append(line)
            mesh_path = root / "dataset_clean" / line / "models" / "model_normalized.obj"
            make_sphere_mesh(str(mesh_path), seed=i)

    rng = np.random.default_rng(123)
    rng.shuffle(all_lines)
    val_lines = all_lines[:10]
    test_lines = all_lines[10:20]
    train_lines = all_lines[20:]

    datalist_dir = root / "datalist"
    for split, lines in (("train", train_lines), ("validation", val_lines), ("test", test_lines)):
        d = datalist_dir / split
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f"{split}.txt", "w") as f:
            f.write("\n".join(lines) + "\n")

    # noisy test
    for line in test_lines:
        noisy_path = root / "test_noisy" / line / "noisy.npy"
        os.makedirs(noisy_path.parent, exist_ok=True)
        pts = np.random.randn(2048, 3).astype(np.float32) * 0.1
        np.save(noisy_path, pts)

    print(f"Synthetic dataset at {root}: train={len(train_lines)} val={len(val_lines)} test={len(test_lines)}")


if __name__ == "__main__":
    main()
