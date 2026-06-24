"""End-to-end smoke test on synthetic data."""
from __future__ import annotations

import os
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def run(cmd, cwd=None, env=None):
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=cwd, env=env)


def main():
    root = Path(tempfile.mkdtemp(prefix="pcdenoise_"))
    print("Using temp root:", root)

    run([sys.executable, "scripts/create_synthetic_data.py", "--root", str(root)], cwd="/workspace")
    run([sys.executable, "scripts/audit_datalist.py", "--datalist_dir", str(root / "datalist")], cwd="/workspace", env={**os.environ, "PYTHONPATH": "/workspace"})

    # Patch configs for tiny fast training
    import yaml

    cfg_path = root / "train_tiny.yaml"
    cfg = {
        "mode": "train",
        "training_stage": "vm",
        "model": {
            "frame_knn": 8,
            "feat_embedding_dim": 64,
            "decoder_hidden_dim": 32,
            "dsm_sigma": 0.01,
            "num_train_points": 256,
            "num_modules": 2,
        },
        "data": {
            "input_dataset_dir": str(root / "dataset_clean"),
            "datalist": str(root / "datalist" / "train" / "train.txt"),
            "use_prob": True,
            "num_files": 4,
            "num_samples": 2048,
            "num_vertex_samples": 0,
            "num_patches": 2,
            "patch_size": 256,
            "batch_size": 2,
        },
        "optimizer": {"lr": 1e-3},
        "trainer": {"epochs": 1, "max_grad_norm": 1.0, "exp_dir": str(root / "exp_vm")},
    }
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f)

    run([sys.executable, "run.py", "--task", str(cfg_path), "--seed", "123"], cwd="/workspace", env={**os.environ, "PYTHONPATH": "/workspace"})
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
