"""Training transforms: noise, patches, pc_mix."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


NOISE_COMPONENTS_JOINT = [
    {"weight": 0.65, "noise_std_min": 0.005, "noise_std_max": 0.020},
    {"weight": 0.35, "noise_std_min": 0.018, "noise_std_max": 0.024},
]


def add_laplace_noise(pc: np.ndarray, noise_std: float) -> np.ndarray:
    noise = np.random.laplace(0.0, noise_std, size=pc.shape)
    return (pc + noise).astype(np.float32)


def sample_noise_std(stage: str) -> float:
    if stage == "joint":
        weights = np.array([c["weight"] for c in NOISE_COMPONENTS_JOINT], dtype=np.float64)
        prob = weights / weights.sum()
        comp = NOISE_COMPONENTS_JOINT[np.random.choice(len(NOISE_COMPONENTS_JOINT), p=prob)]
        return float(np.random.uniform(comp["noise_std_min"], comp["noise_std_max"]))
    return float(np.random.uniform(0.005, 0.020))


def make_training_patches(
    pc_clean: np.ndarray,
    pc_noisy: np.ndarray,
    normals: np.ndarray,
    num_patches: int = 4,
    patch_size: int = 1000,
    shared_time: bool = True,
):
    n = pc_noisy.shape[0]
    num_patches = min(num_patches, n)
    patch_size = min(patch_size, n)
    seed_idx = np.random.permutation(n)[:num_patches]
    seed_points = pc_noisy[seed_idx]

    tree = cKDTree(pc_noisy)
    _, nn_idx = tree.query(seed_points, k=patch_size)
    if nn_idx.ndim == 1:
        nn_idx = nn_idx.reshape(1, -1)

    pat_a = pc_noisy[nn_idx]
    pat_b = pc_clean[nn_idx]
    pat_n = normals[nn_idx]

    if shared_time:
        patch_time = np.random.rand(num_patches, 1, 1).astype(np.float32)
    else:
        patch_time = np.random.rand(num_patches, 1, 1).astype(np.float32)

    t = np.broadcast_to(patch_time, (num_patches, patch_size, 1))
    t = (1.0 - 1e-8) * t + 1e-8

    pat_t = t * pat_b + (1.0 - t) * pat_a
    seed_points_t = (
        t[:, 0:1, :] * pc_clean[seed_idx][:, None, :]
        + (1.0 - t[:, 0:1, :]) * pc_noisy[seed_idx][:, None, :]
    )

    pat_a = (pat_a - seed_points_t).astype(np.float32)
    pat_b = (pat_b - seed_points_t).astype(np.float32)
    pat_t = (pat_t - seed_points_t).astype(np.float32)

    return {
        "pc_noisy": pat_a,
        "pc_clean": pat_b,
        "pc_mix": pat_t,
        "surface_normals": pat_n.astype(np.float32),
        "flow_time": patch_time.reshape(num_patches, 1).astype(np.float32),
    }


def random_train_indices(num_points: int, num_train_points: int) -> np.ndarray:
    if num_points <= num_train_points:
        return np.arange(num_points)
    return np.random.choice(num_points, num_train_points, replace=False)
