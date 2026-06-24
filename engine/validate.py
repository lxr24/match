"""Validation scoring on cached noisy/clean pairs."""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

from data.path_resolver import read_datalist, resolve_noisy_path, resolve_val_clean_path, resolve_val_noisy_path
from engine.predict import denoise_point_cloud, load_model_from_checkpoint
from metrics.evaluate import evaluate_samples


def run_validation(
    cfg: dict,
    ckpt_path: str,
    list_file: str,
    val_cache_root: str,
    mesh_dir: str,
    pred_dir: str,
    max_samples: Optional[int] = None,
) -> Dict[str, float]:
    os.makedirs(pred_dir, exist_ok=True)
    lines = read_datalist(list_file)
    if max_samples is not None:
        lines = lines[:max_samples]

    model = load_model_from_checkpoint(cfg, ckpt_path)
    pred_cfg = cfg.get("predict", {})

    for line in lines:
        noisy_path = resolve_val_noisy_path(val_cache_root, line)
        points = np.load(noisy_path).astype(np.float32)
        denoised = denoise_point_cloud(
            model,
            points,
            patch_size=pred_cfg.get("predict_patch_size", 1000),
            seed_k=pred_cfg.get("predict_seed_k", 6),
            inner_steps=pred_cfg.get("predict_inner_steps", 2),
            step_scale=pred_cfg.get("predict_step_scale", 1.0),
        )
        out_path = os.path.join(pred_dir, line, "denoised.npy")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.save(out_path, denoised.astype(np.float32))

    metrics = evaluate_samples(
        pred_dir=pred_dir,
        gt_dir=os.path.join(val_cache_root, "clean"),
        noisy_dir=os.path.join(val_cache_root, "noisy"),
        mesh_dir=mesh_dir,
        list_file=list_file,
    )
    return metrics
