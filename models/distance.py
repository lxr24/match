"""Patch-level distance ratio module."""
from __future__ import annotations

import jittor as jt
from jittor import nn

from models.edgeconv import StraightFeatureExtraction


class PatchDistanceModule(nn.Module):
    def __init__(
        self,
        frame_knn: int = 32,
        distance_embedding_dim: int = 128,
        distance_hidden_dim: int = 64,
        feat_embedding_dim: int = 256,
    ):
        super().__init__()
        self.encoder = StraightFeatureExtraction(frame_knn, 3, distance_embedding_dim)
        self.head = nn.Sequential(
            nn.Linear(distance_embedding_dim, distance_hidden_dim),
            nn.ReLU(),
            nn.Linear(distance_hidden_dim, 1),
        )

    def execute(self, x: jt.Var) -> jt.Var:
        # x: (B, N, 3) -> ratio (B, 1, 1)
        feat = self.encoder(x)
        pooled = feat.max(dim=1)  # (B, C)
        ratio = jt.sigmoid(self.head(pooled))
        return ratio.reshape(-1, 1, 1)
