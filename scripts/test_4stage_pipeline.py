"""Quick 4-stage pipeline test on synthetic data."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def run(cmd, env):
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT), env=env)


def tiny_cfg(root: Path, stage: str, init_ckpt: str | None, exp: str) -> Path:
    model = {
        "frame_knn": 8,
        "feat_embedding_dim": 64,
        "decoder_hidden_dim": 32,
        "dsm_sigma": 0.01,
        "num_train_points": 128,
        "num_modules": 2,
        "distance_embedding_dim": 64,
        "distance_hidden_dim": 32,
        "distance_ratio_loss_weight": 1.0,
        "distance_final_loss_weight": 200.0,
        "cvm_dir_loss_weight": 1.0,
        "cvm_consistency_loss_weight": 10.0,
        "use_coverage_loss": stage == "joint",
        "use_mesh_loss": stage == "joint",
        "coverage_loss_weight": 75.0,
        "coverage_pred_weight": 0.25,
        "coverage_clean_weight": 0.75,
        "coverage_num_points": 64,
        "mesh_loss_weight": 0.05,
    }
    cfg = {
        "mode": "train",
        "training_stage": stage,
        "model": model,
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
        "optimizer": {"lr": 1e-3 if stage != "joint" else 2e-4},
        "trainer": {"epochs": 1, "max_grad_norm": 1.0, "exp_dir": str(root / exp)},
    }
    if init_ckpt:
        cfg["init_ckpt"] = init_ckpt
    path = root / f"cfg_{stage}.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f)
    return path


def main():
    root = Path(tempfile.mkdtemp(prefix="pcd4_"))
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    run([sys.executable, "scripts/create_synthetic_data.py", "--root", str(root)], env)

    vm_cfg = tiny_cfg(root, "vm", None, "exp_vm")
    run([sys.executable, "run.py", "--task", str(vm_cfg), "--seed", "123"], env)
    vm_ckpt = root / "exp_vm" / "checkpoint_0.pkl"

    cvm_cfg = tiny_cfg(root, "cvm", str(vm_ckpt), "exp_cvm")
    run([sys.executable, "run.py", "--task", str(cvm_cfg), "--seed", "123"], env)
    cvm_ckpt = root / "exp_cvm" / "checkpoint_0.pkl"

    core_cfg = tiny_cfg(root, "distance", str(cvm_ckpt), "exp_core")
    run([sys.executable, "run.py", "--task", str(core_cfg), "--seed", "123"], env)
    core_ckpt = root / "exp_core" / "checkpoint_0.pkl"

    joint_cfg = tiny_cfg(root, "joint", str(core_ckpt), "exp_joint")
    run([sys.executable, "run.py", "--task", str(joint_cfg), "--seed", "123"], env)

    # build val cache + predict one sample
    run([
        sys.executable, "scripts/build_val_cache.py",
        "--dataset_root", str(root / "dataset_clean"),
        "--datalist_dir", str(root / "datalist"),
        "--val_cache_root", str(root / "val_cache"),
        "--num_points", "2048",
    ], env)

    joint_ckpt = root / "exp_joint" / "checkpoint_0.pkl"
    val_cfg = {
        "mode": "validate",
        "datalist_dir": str(root / "datalist"),
        "val_cache_root": str(root / "val_cache"),
        "mesh_dir": str(root / "dataset_clean"),
        "pred_dir": str(root / "val_pred"),
        "model": yaml.safe_load(open(joint_cfg))["model"],
        "predict": {"predict_patch_size": 256, "predict_seed_k": 4, "predict_inner_steps": 1},
    }
    val_path = root / "validate.yaml"
    with open(val_path, "w") as f:
        yaml.dump(val_cfg, f)

    run([
        sys.executable, "run.py", "--task", str(val_path),
        "--checkpoint", str(joint_ckpt),
    ], env)

    print("4-stage pipeline test passed.")


if __name__ == "__main__":
    main()
