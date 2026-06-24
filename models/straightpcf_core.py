"""StraightPCFCore model with stage-specific losses and inference."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import jittor as jt
from jittor import nn

from models.cvm import CoupledVelocityModule
from models.distance import PatchDistanceModule
from models.velocity import StraightVelocityModule


class StraightPCFCore(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.training_stage = cfg.get("training_stage", "vm")
        self.num_modules = cfg.get("num_modules", 2)
        self.dsm_sigma = cfg.get("dsm_sigma", 0.01)
        self.num_train_points = cfg.get("num_train_points", 1000)
        self.cvm_dir_loss_weight = cfg.get("cvm_dir_loss_weight", 1.0)
        self.cvm_consistency_loss_weight = cfg.get("cvm_consistency_loss_weight", 10.0)
        self.distance_ratio_loss_weight = cfg.get("distance_ratio_loss_weight", 1.0)
        self.distance_final_loss_weight = cfg.get("distance_final_loss_weight", 200.0)
        self.use_coverage_loss = cfg.get("use_coverage_loss", False)
        self.use_mesh_loss = cfg.get("use_mesh_loss", False)
        self.coverage_loss_weight = cfg.get("coverage_loss_weight", 75.0)
        self.coverage_pred_weight = cfg.get("coverage_pred_weight", 0.25)
        self.coverage_clean_weight = cfg.get("coverage_clean_weight", 0.75)
        self.coverage_num_points = cfg.get("coverage_num_points", 512)
        self.mesh_loss_weight = cfg.get("mesh_loss_weight", 0.05)

        self.single_velocity = StraightVelocityModule(
            frame_knn=cfg.get("frame_knn", 32),
            feat_embedding_dim=cfg.get("feat_embedding_dim", 256),
            decoder_hidden_dim=cfg.get("decoder_hidden_dim", 64),
        )
        self.cvm = CoupledVelocityModule(cfg, self.num_modules)
        self.distance = PatchDistanceModule(
            frame_knn=cfg.get("frame_knn", 32),
            distance_embedding_dim=cfg.get("distance_embedding_dim", 128),
            distance_hidden_dim=cfg.get("distance_hidden_dim", 64),
            feat_embedding_dim=cfg.get("feat_embedding_dim", 256),
        )

    def set_stage(self, stage: str):
        self.training_stage = stage

    def _random_indices(self, num_points: int) -> jt.Var:
        n = min(num_points, self.num_train_points)
        idx = jt.randperm(num_points)[:n]
        return idx

    def mse_vector_loss(self, pred: jt.Var, target: jt.Var, indices: Optional[jt.Var] = None) -> jt.Var:
        if indices is not None:
            pred = pred[:, indices, :]
            target = target[:, indices, :]
        return ((pred - target) ** 2).mean()

    def get_vm_loss(self, pc_noisy, pc_mix, pc_clean) -> jt.Var:
        pred_delta = self.single_velocity.decode_delta(pc_mix)
        target_delta = pc_clean - pc_noisy
        idx = self._random_indices(pc_mix.shape[1])
        loss = self.mse_vector_loss(pred_delta, target_delta, idx)
        return loss / self.dsm_sigma

    def get_cvm_loss(self, pc_noisy, pc_mix, pc_clean, flow_time) -> jt.Var:
        idx = self._random_indices(pc_mix.shape[1])
        target_delta = pc_clean - pc_noisy
        current = pc_mix
        flow = flow_time.reshape(-1, 1, 1)
        step_scale = (1.0 - flow) / self.num_modules

        total_dir = jt.array(0.0)
        total_cons = jt.array(0.0)

        for module_id, module in enumerate(self.cvm.velocity_nets):
            pred_delta = module.decode_delta(current)
            total_dir += self.mse_vector_loss(pred_delta, target_delta, idx)
            current = current + step_scale * pred_delta

            if module_id < self.num_modules - 1:
                next_id = module_id + 1
                next_t = (flow_time * (self.num_modules - next_id) + next_id) / self.num_modules
                next_t = next_t.reshape(-1, 1, 1)
                target_current = next_t * pc_clean + (1.0 - next_t) * pc_noisy
                total_cons += self.mse_vector_loss(current, target_current, idx)

        loss = (
            self.cvm_dir_loss_weight * total_dir
            + self.cvm_consistency_loss_weight * total_cons
        )
        return loss / self.dsm_sigma

    def cvm_distance_update(
        self,
        current: jt.Var,
        ratio: jt.Var,
        num_steps: int = 1,
        step_scale: float = 1.0,
    ) -> jt.Var:
        current_next = current
        for _ in range(num_steps):
            for module in self.cvm.velocity_nets:
                pred_delta = module.decode_delta(current_next)
                current_next = current_next + (
                    step_scale / num_steps / self.num_modules
                ) * ratio * pred_delta
        return current_next

    def distance_target_ratio(self, pc_noisy, pc_mix, pc_clean, flow_time) -> jt.Var:
        if flow_time is not None:
            return (1.0 - flow_time).reshape(-1, 1, 1)
        num = jt.norm(pc_clean - pc_mix, dim=-1).mean(dim=1, keepdims=True)
        den = jt.norm(pc_clean - pc_noisy, dim=-1).mean(dim=1, keepdims=True)
        return num / (den + 1e-6)

    def get_distance_loss(self, pc_noisy, pc_mix, pc_clean, flow_time) -> Tuple[jt.Var, jt.Var]:
        ratio = self.distance(pc_mix)
        target_ratio = self.distance_target_ratio(pc_noisy, pc_mix, pc_clean, flow_time)
        ratio_loss = ((ratio - target_ratio) ** 2).mean()
        pc_pred = self.cvm_distance_update(pc_mix, ratio)
        final_loss = ((pc_pred - pc_clean) ** 2).mean()
        loss = (
            self.distance_ratio_loss_weight * ratio_loss
            + self.distance_final_loss_weight * final_loss
        ) / self.dsm_sigma
        return loss, pc_pred

    def balanced_coverage_loss(self, pc_pred, pc_clean) -> jt.Var:
        b, n, _ = pc_pred.shape
        k = min(self.coverage_num_points, n)
        pred_idx = jt.randperm(n)[:k]
        clean_idx = jt.randperm(n)[:k]
        pred_sub = pc_pred[:, pred_idx, :]
        clean_sub = pc_clean[:, clean_idx, :]

        dist = ((pred_sub.unsqueeze(2) - clean_sub.unsqueeze(1)) ** 2).sum(-1)
        pred_to_clean = dist.min(dim=2).mean()
        clean_to_pred = dist.min(dim=1).mean()
        return (
            self.coverage_pred_weight * pred_to_clean
            + self.coverage_clean_weight * clean_to_pred
        ) / self.dsm_sigma

    def mesh_normal_loss(self, pc_pred, pc_clean, surface_normals) -> jt.Var:
        idx = self._random_indices(pc_pred.shape[1])
        pred = pc_pred[:, idx, :]
        clean = pc_clean[:, idx, :]
        normals = surface_normals[:, idx, :]
        norm_len = jt.norm(normals, dim=-1, keepdims=True) + 1e-8
        normals = normals / norm_len
        normal_error = ((pred - clean) * normals).sum(dim=-1)
        return ((normal_error ** 2).mean()) / self.dsm_sigma

    def get_joint_loss(self, pc_noisy, pc_mix, pc_clean, surface_normals, flow_time) -> jt.Var:
        dist_loss, pc_pred = self.get_distance_loss(pc_noisy, pc_mix, pc_clean, flow_time)
        loss = dist_loss
        if self.use_coverage_loss:
            loss = loss + self.coverage_loss_weight * self.balanced_coverage_loss(pc_pred, pc_clean)
        if self.use_mesh_loss:
            loss = loss + self.mesh_loss_weight * self.mesh_normal_loss(pc_pred, pc_clean, surface_normals)
        return loss

    def compute_loss(self, batch: Dict[str, jt.Var]) -> jt.Var:
        pc_noisy = batch["pc_noisy"]
        pc_mix = batch["pc_mix"]
        pc_clean = batch["pc_clean"]
        surface_normals = batch.get("surface_normals")
        flow_time = batch["flow_time"]

        stage = self.training_stage
        if stage == "vm":
            return self.get_vm_loss(pc_noisy, pc_mix, pc_clean)
        if stage == "cvm":
            return self.get_cvm_loss(pc_noisy, pc_mix, pc_clean, flow_time)
        if stage == "distance":
            loss, _ = self.get_distance_loss(pc_noisy, pc_mix, pc_clean, flow_time)
            return loss
        if stage == "joint":
            return self.get_joint_loss(pc_noisy, pc_mix, pc_clean, surface_normals, flow_time)
        raise ValueError(f"Unknown training stage: {stage}")

    def denoise_patch(
        self,
        patch: jt.Var,
        num_steps: int = 2,
        step_scale: float = 1.0,
    ) -> jt.Var:
        current = patch
        ratio = self.distance(current)
        return self.cvm_distance_update(current, ratio, num_steps=num_steps, step_scale=step_scale)

    def execute(self, batch: Dict[str, jt.Var]) -> jt.Var:
        return self.compute_loss(batch)
