"""Predict on test_noisy using datalist/test."""
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
from data.path_resolver import find_datalist_file, read_datalist, resolve_noisy_path, resolve_output_path
from engine.predict import denoise_point_cloud, load_model_from_checkpoint
from utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test_root", default="./test_noisy")
    parser.add_argument("--output_root", default="./results")
    parser.add_argument("--datalist_dir", default="./datalist")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.task)
    model = load_model_from_checkpoint(cfg, args.checkpoint)
    pred_cfg = cfg.get("predict", {})

    list_file = find_datalist_file(args.datalist_dir, "test")
    lines = read_datalist(list_file)
    if args.max_samples:
        lines = lines[: args.max_samples]

    for line in tqdm(lines, desc="predict_test"):
        noisy_path = resolve_noisy_path(args.test_root, line)
        points = np.load(noisy_path).astype(np.float32)
        denoised = denoise_point_cloud(
            model,
            points,
            patch_size=pred_cfg.get("predict_patch_size", 1000),
            seed_k=pred_cfg.get("predict_seed_k", 6),
            inner_steps=pred_cfg.get("predict_inner_steps", 2),
            step_scale=pred_cfg.get("predict_step_scale", 1.0),
        )
        assert denoised.shape == points.shape, f"shape mismatch for {line}"
        out_path = resolve_output_path(args.output_root, line)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.save(out_path, denoised.astype(np.float32))

    print(f"Wrote {len(lines)} predictions to {args.output_root}")


if __name__ == "__main__":
    main()
