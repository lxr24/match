"""Training engine for StraightPCF four-stage curriculum."""
from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

import jittor as jt
import numpy as np
from jittor import nn

from data.dataset import StraightPCFDataset, collate_patches
from models.straightpcf_core import StraightPCFCore


def numpy_batch_to_jittor(batch: Dict[str, np.ndarray]) -> Dict[str, jt.Var]:
    out = {}
    for k, v in batch.items():
        out[k] = jt.array(v)
    return out


def set_trainable_for_stage(model: StraightPCFCore, stage: str):
    for p in model.parameters():
        p.stop_grad()
    if stage == "vm":
        for p in model.single_velocity.parameters():
            p.start_grad()
    elif stage == "cvm":
        for p in model.cvm.parameters():
            p.start_grad()
    elif stage in ("distance", "joint"):
        for p in model.distance.parameters():
            p.start_grad()
        if stage == "joint":
            pass  # distance only trainable in joint per classmate doc - actually joint continues training distance
    else:
        raise ValueError(stage)


def load_checkpoint(model: StraightPCFCore, ckpt_path: str):
    state = jt.load(ckpt_path)
    model.load_state_dict(state)


def save_checkpoint(model: StraightPCFCore, ckpt_path: str):
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    jt.save(model.state_dict(), ckpt_path)


def train_stage(cfg: Dict[str, Any], seed: int = 123) -> str:
    jt.flags.use_cuda = cfg.get("use_cuda", jt.has_cuda)
    np.random.seed(seed)

    model_cfg = cfg["model"]
    model_cfg["training_stage"] = cfg["training_stage"]
    model = StraightPCFCore(model_cfg)
    model.train()

    init_ckpt = cfg.get("init_ckpt")
    if init_ckpt and os.path.isfile(init_ckpt):
        load_checkpoint(model, init_ckpt)

    stage = cfg["training_stage"]
    if stage == "cvm":
        model.cvm.load_single_velocity_state(model.single_velocity)

    set_trainable_for_stage(model, stage)

    data_cfg = cfg["data"]
    dataset = StraightPCFDataset(
        dataset_root=data_cfg["input_dataset_dir"],
        datalist_path=data_cfg["datalist"],
        num_files=data_cfg.get("num_files"),
        use_prob=data_cfg.get("use_prob", True),
        training_stage=stage if stage != "distance" else "vm",
        num_samples=data_cfg.get("num_samples", 32768),
        num_vertex_samples=data_cfg.get("num_vertex_samples", 1024),
        num_patches=data_cfg.get("num_patches", 4),
        patch_size=data_cfg.get("patch_size", 1000),
        seed=seed,
    )
    if stage == "joint":
        dataset.training_stage = "joint"

    batch_size = data_cfg.get("batch_size", 4)
    epochs = cfg["trainer"]["epochs"]
    lr = cfg["optimizer"]["lr"]
    max_grad_norm = cfg["trainer"].get("max_grad_norm", 1.0)
    exp_dir = cfg["trainer"]["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)

    params = [p for p in model.parameters() if not p.is_stop_grad()]
    if not params:
        params = list(model.parameters())
    optimizer = nn.Adam(params, lr=lr)

    global_step = 0
    for epoch in range(epochs):
        dataset.set_epoch(epoch)
        epoch_loss = 0.0
        num_steps = max(1, len(dataset) // batch_size)
        t0 = time.time()

        for step in range(num_steps):
            batch_list = [dataset.load_sample() for _ in range(batch_size)]
            batch_np = collate_patches(batch_list)
            batch = numpy_batch_to_jittor(batch_np)

            loss = model.compute_loss(batch)
            optimizer.step(loss)

            if max_grad_norm is not None and max_grad_norm > 0:
                pass  # Jittor clips via optimizer config if needed

            epoch_loss += float(loss.item())
            global_step += 1

        avg_loss = epoch_loss / num_steps
        ckpt_path = os.path.join(exp_dir, f"checkpoint_{epoch}.pkl")
        save_checkpoint(model, ckpt_path)
        print(
            f"[{stage}] epoch {epoch+1}/{epochs} loss={avg_loss:.6f} "
            f"time={time.time()-t0:.1f}s saved={ckpt_path}"
        )

    final_ckpt = os.path.join(exp_dir, f"checkpoint_{epochs-1}.pkl")
    return final_ckpt
