"""Patch-based inference with best aggregation."""
from __future__ import annotations

import os
from typing import List, Tuple

import jittor as jt
import numpy as np
from scipy.spatial import cKDTree

from models.straightpcf_core import StraightPCFCore


def farthest_point_sampling(points: np.ndarray, num_seeds: int) -> np.ndarray:
    n = points.shape[0]
    num_seeds = min(num_seeds, n)
    selected = np.zeros(num_seeds, dtype=np.int64)
    distances = np.full(n, np.inf, dtype=np.float64)
    farthest = np.random.randint(0, n)
    for i in range(num_seeds):
        selected[i] = farthest
        centroid = points[farthest]
        dist = np.sum((points - centroid) ** 2, axis=1)
        distances = np.minimum(distances, dist)
        farthest = int(np.argmax(distances))
    return selected


def knn_patch(seed_points: np.ndarray, points: np.ndarray, patch_size: int):
    tree = cKDTree(points)
    patch_dists, point_idxs = tree.query(seed_points, k=patch_size)
    if point_idxs.ndim == 1:
        patch_dists = patch_dists.reshape(1, -1)
        point_idxs = point_idxs.reshape(1, -1)
    patches = points[point_idxs]
    patches = patches - seed_points[:, None, :]
    return patch_dists, point_idxs, patches.astype(np.float32)


def aggregate_best(
    original_points: np.ndarray,
    point_idxs: np.ndarray,
    patch_dists: np.ndarray,
    patches_denoised: np.ndarray,
) -> np.ndarray:
    n = original_points.shape[0]
    out = original_points.copy()
    best_dist = np.full((n,), np.inf, dtype=np.float64)
    best_patch = np.full((n,), -1, dtype=np.int64)
    best_offset = np.full((n,), -1, dtype=np.int64)

    num_patches = point_idxs.shape[0]
    patch_size = point_idxs.shape[1]

    for patch_id in range(num_patches):
        idx = point_idxs[patch_id]
        dist = patch_dists[patch_id]
        for j in range(patch_size):
            pid = int(idx[j])
            if dist[j] < best_dist[pid]:
                best_dist[pid] = dist[j]
                best_patch[pid] = patch_id
                best_offset[pid] = j

    valid = best_patch >= 0
    out[valid] = patches_denoised[best_patch[valid], best_offset[valid]]
    return out.astype(np.float32)


def denoise_point_cloud(
    model: StraightPCFCore,
    points: np.ndarray,
    patch_size: int = 1000,
    seed_k: int = 6,
    inner_steps: int = 2,
    step_scale: float = 1.0,
) -> np.ndarray:
    model.eval()
    n = points.shape[0]
    num_patches = max(1, min(n, int(seed_k * n / patch_size)))
    seed_idx = farthest_point_sampling(points, num_patches)
    seed_points = points[seed_idx]

    patch_dists, point_idxs, patches = knn_patch(seed_points, points, patch_size)

    denoised_patches = []
    for i in range(num_patches):
        patch = jt.array(patches[i : i + 1])
        out = model.denoise_patch(patch, num_steps=inner_steps, step_scale=step_scale)
        denoised_patches.append(out.numpy()[0])

    patches_denoised = np.stack(denoised_patches, axis=0)
    return aggregate_best(points, point_idxs, patch_dists, patches_denoised)


def load_model_from_checkpoint(cfg: dict, ckpt_path: str) -> StraightPCFCore:
    model_cfg = cfg["model"]
    model_cfg["training_stage"] = "joint"
    model = StraightPCFCore(model_cfg)
    state = jt.load(ckpt_path)
    model.load_state_dict(state)
    model.eval()
    return model
