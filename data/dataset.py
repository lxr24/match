"""Online mesh dataset for StraightPCF training."""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

from data.mesh_utils import normalize_pc, sample_mesh_surface, load_mesh
from data.path_resolver import read_datalist, resolve_mesh_path
from data.transform import add_laplace_noise, make_training_patches, sample_noise_std


class StraightPCFDataset:
    def __init__(
        self,
        dataset_root: str,
        datalist_path: str,
        num_files: Optional[int] = None,
        use_prob: bool = True,
        training_stage: str = "vm",
        num_samples: int = 32768,
        num_vertex_samples: int = 1024,
        num_patches: int = 4,
        patch_size: int = 1000,
        shuffle: bool = True,
        seed: int = 123,
    ):
        self.dataset_root = dataset_root
        self.lines = read_datalist(datalist_path)
        self.training_stage = training_stage
        self.num_samples = num_samples
        self.num_vertex_samples = num_vertex_samples
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.shuffle = shuffle
        self.seed = seed
        self.use_prob = use_prob
        self.epoch_size = num_files if num_files is not None else len(self.lines)
        self.rng = np.random.default_rng(seed)

        if len(self.lines) == 0:
            raise ValueError(f"Empty datalist: {datalist_path}")

    def __len__(self) -> int:
        return self.epoch_size

    def _sample_line(self) -> str:
        if self.use_prob:
            idx = self.rng.integers(0, len(self.lines))
        else:
            idx = self.rng.integers(0, len(self.lines))
        return self.lines[idx]

    def load_sample(self) -> Dict[str, np.ndarray]:
        line = self._sample_line()
        mesh_path = resolve_mesh_path(self.dataset_root, line)
        if not os.path.isfile(mesh_path):
            raise FileNotFoundError(mesh_path)

        vertices, faces = load_mesh(mesh_path)
        pc_clean, normals = sample_mesh_surface(
            vertices, faces, self.num_samples, self.num_vertex_samples
        )
        pc_clean = normalize_pc(pc_clean)
        noise_std = sample_noise_std(self.training_stage)
        pc_noisy = add_laplace_noise(pc_clean, noise_std)

        patches = make_training_patches(
            pc_clean,
            pc_noisy,
            normals,
            num_patches=self.num_patches,
            patch_size=self.patch_size,
            shared_time=True,
        )
        return patches

    def set_epoch(self, epoch: int):
        self.rng = np.random.default_rng(self.seed + epoch)


def collate_patches(batch: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Flatten list of patch dicts into batched arrays (P_total, M, 3)."""
    keys = ["pc_noisy", "pc_clean", "pc_mix", "surface_normals"]
    out = {}
    for key in keys:
        out[key] = np.concatenate([b[key] for b in batch], axis=0).astype(np.float32)
    out["flow_time"] = np.concatenate([b["flow_time"] for b in batch], axis=0).astype(np.float32)
    return out
