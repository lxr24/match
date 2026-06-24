"""Coupled velocity modules."""
from __future__ import annotations

from typing import List

import jittor as jt
from jittor import nn

from models.velocity import StraightVelocityModule


class CoupledVelocityModule(nn.Module):
    def __init__(self, cfg: dict, num_modules: int = 2):
        super().__init__()
        self.num_modules = num_modules
        self.velocity_nets = nn.ModuleList(
            [
                StraightVelocityModule(
                    frame_knn=cfg.get("frame_knn", 32),
                    feat_embedding_dim=cfg.get("feat_embedding_dim", 256),
                    decoder_hidden_dim=cfg.get("decoder_hidden_dim", 64),
                )
                for _ in range(num_modules)
            ]
        )

    def load_single_velocity_state(self, velocity_module: StraightVelocityModule):
        state = velocity_module.state_dict()
        for module in self.velocity_nets:
            module.load_state_dict(state)

    def execute(self, x: jt.Var) -> jt.Var:
        return self.velocity_nets[0].decode_delta(x)
