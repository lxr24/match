"""Unified entry point."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="StraightPCF point cloud denoising")
    parser.add_argument("--task", required=True, help="Path to task yaml")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    cfg = load_config(args.task)
    mode = cfg.get("mode", "train")

    if mode == "train":
        from engine.train import train_stage

        cfg["training_stage"] = cfg.get("training_stage", "vm")
        ckpt = train_stage(cfg, seed=args.seed)
        print(f"Training finished. Last checkpoint: {ckpt}")
        return

    if mode == "validate":
        from engine.validate import run_validation
        from data.path_resolver import find_datalist_file

        metrics = run_validation(
            cfg=cfg,
            ckpt_path=args.checkpoint or cfg["checkpoint"],
            list_file=cfg.get("list_file") or find_datalist_file(cfg.get("datalist_dir", "./datalist"), "validation"),
            val_cache_root=cfg.get("val_cache_root", "./val_cache"),
            mesh_dir=cfg.get("mesh_dir", "./dataset_clean"),
            pred_dir=cfg.get("pred_dir", "./outputs/validation_predictions"),
            max_samples=cfg.get("max_samples"),
        )
        print(
            f"samples={metrics['samples']} score={metrics['score']:.4f} "
            f"cd_score={metrics['cd_score']:.4f} p2s_score={metrics['p2s_score']:.4f}"
        )
        return

    if mode == "predict":
        from scripts.predict_test import main as predict_main

        sys.argv = [
            "predict_test.py",
            "--task",
            args.task,
            "--checkpoint",
            args.checkpoint or cfg["checkpoint"],
        ]
        predict_main()
        return

    raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
