"""Velocity module for displacement prediction."""
from __future__ import annotations

import jittor as jt
from jittor import nn

from models.edgeconv import StraightFeatureExtraction


class StraightVelocityModule(nn.Module):
    def __init__(
        self,
        frame_knn: int = 32,
        feat_embedding_dim: int = 256,
        decoder_hidden_dim: int = 64,
    ):
        super().__init__()
        self.encoder = StraightFeatureExtraction(frame_knn, 3, feat_embedding_dim)
        self.decoder = nn.Sequential(
            nn.Linear(feat_embedding_dim, decoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(decoder_hidden_dim, 3),
        )

    def encode(self, x: jt.Var) -> jt.Var:
        return self.encoder(x)

    def decode_delta(self, x: jt.Var) -> jt.Var:
        feat = self.encoder(x)
        return self.decoder(feat)

    def execute(self, x: jt.Var) -> jt.Var:
        return self.decode_delta(x)
