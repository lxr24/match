"""Evaluation metrics aligned with competition scoring."""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

from data.path_resolver import read_datalist, resolve_mesh_path


def normalize_point_cloud(pc: np.ndarray) -> np.ndarray:
    center = pc.mean(axis=0)
    pc = pc - center
    scale = np.sqrt((pc ** 2).sum(axis=1).max())
    scale = max(float(scale), 1e-8)
    return pc / scale


def chamfer_distance(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.spatial import cKDTree

    ta = cKDTree(a)
    tb = cKDTree(b)
    da, _ = tb.query(a, k=1)
    db, _ = ta.query(b, k=1)
    return float(da.mean() + db.mean())


def point_to_surface_distance(pred: np.ndarray, mesh_path: str) -> Optional[float]:
    try:
        import point_cloud_utils as pcu
    except ImportError:
        return None

    import trimesh

    mesh = trimesh.load(mesh_path, process=False, force="mesh")
    if hasattr(mesh, "vertices"):
        v = np.asarray(mesh.vertices, dtype=np.float64)
        f = np.asarray(mesh.faces, dtype=np.int32)
    else:
        return None
    dist, _ = pcu.closest_points_on_mesh(pred.astype(np.float64), v, f)
    return float(np.mean(dist ** 2))


def score_ratio(pred_metric: float, noisy_metric: float) -> float:
    if noisy_metric <= 1e-12:
        return 0.0
    return float(np.clip(100.0 * (1.0 - pred_metric / noisy_metric), 0.0, 100.0))


def evaluate_samples(
    pred_dir: str,
    gt_dir: str,
    noisy_dir: str,
    mesh_dir: str,
    list_file: str,
    workers: int = 4,
) -> Dict[str, float]:
    lines = read_datalist(list_file)
    cd_scores: List[float] = []
    p2s_scores: List[float] = []
    has_p2s = True

    for line in lines:
        pred_path = os.path.join(pred_dir, line, "denoised.npy")
        gt_path = os.path.join(gt_dir, line, "clean.npy")
        noisy_path = os.path.join(noisy_dir, line, "noisy.npy")
        mesh_path = resolve_mesh_path(mesh_dir, line)

        pred = np.load(pred_path).astype(np.float32)
        gt = np.load(gt_path).astype(np.float32)
        noisy = np.load(noisy_path).astype(np.float32)

        gt_n = normalize_point_cloud(gt)
        pred_n = normalize_point_cloud(pred)
        noisy_n = normalize_point_cloud(noisy)

        cd_pred = chamfer_distance(pred_n, gt_n)
        cd_noisy = chamfer_distance(noisy_n, gt_n)
        cd_scores.append(score_ratio(cd_pred, cd_noisy))

        p2s_pred = point_to_surface_distance(pred_n, mesh_path)
        p2s_noisy = point_to_surface_distance(noisy_n, mesh_path)
        if p2s_pred is None or p2s_noisy is None:
            has_p2s = False
        else:
            p2s_scores.append(score_ratio(p2s_pred, p2s_noisy))

    result = {
        "samples": len(lines),
        "cd_score": float(np.mean(cd_scores)) if cd_scores else 0.0,
        "p2s_score": float(np.mean(p2s_scores)) if p2s_scores else 0.0,
    }
    if has_p2s and p2s_scores:
        result["score"] = 0.5 * result["cd_score"] + 0.5 * result["p2s_score"]
    else:
        result["score"] = result["cd_score"]
    return result


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--noisy_dir", required=True)
    parser.add_argument("--mesh_dir", default=None)
    parser.add_argument("--list_file", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    metrics = evaluate_samples(
        pred_dir=args.pred_dir,
        gt_dir=args.gt_dir,
        noisy_dir=args.noisy_dir,
        mesh_dir=args.mesh_dir or ".",
        list_file=args.list_file,
        workers=args.workers,
    )
    print(
        f"samples={metrics['samples']} score={metrics['score']:.4f} "
        f"cd_score={metrics['cd_score']:.4f} p2s_score={metrics['p2s_score']:.4f}"
    )


if __name__ == "__main__":
    main()
