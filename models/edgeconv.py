"""DGCNN-style EdgeConv feature extraction."""
from __future__ import annotations

import jittor as jt
from jittor import nn


def knn(x: jt.Var, k: int) -> jt.Var:
    """x: (B, N, C) -> idx: (B, N, k)"""
    inner = -2 * nn.matmul(x, x.transpose(0, 2, 1))
    xx = jt.sum(x * x, dim=-1, keepdims=True)
    pairwise = -xx - inner - xx.transpose(0, 2, 1)
    sorted_idx = jt.argsort(pairwise, dim=-1)
    if isinstance(sorted_idx, tuple):
        sorted_idx = sorted_idx[0]
    return sorted_idx[:, :, :k]


def index_points(points: jt.Var, idx: jt.Var) -> jt.Var:
    # points: (B, N, C), idx: (B, M, K)
    b, n, c = points.shape
    _, m, k = idx.shape
    flat_idx = idx.reshape(b, m * k)
    batch_idx = jt.arange(b).reshape(b, 1).repeat(1, m * k)
    gathered = points[batch_idx, flat_idx]
    return gathered.reshape(b, m, k, c)


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = None):
        super().__init__()
        hidden = hidden or out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def execute(self, x):
        return self.net(x)


class StraightEdgeConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.mlp = MLP(2 * in_channels, out_channels, hidden=out_channels)
        self.lin = nn.Linear(in_channels, out_channels)

    def execute(self, x: jt.Var, knn_idx: jt.Var) -> jt.Var:
        neighbors = index_points(x, knn_idx)
        x_i = x.unsqueeze(2).repeat(1, 1, knn_idx.shape[2], 1)
        edge_input = jt.concat([x_i, neighbors - x_i], dim=-1)
        msg = self.mlp(edge_input)
        out = msg.max(dim=2)
        return out + self.lin(x)


class StraightFeatureExtraction(nn.Module):
    def __init__(self, k: int = 32, input_dim: int = 3, embedding_dim: int = 256):
        super().__init__()
        self.k = k
        self.conv1 = StraightEdgeConv(input_dim, embedding_dim // 8)
        self.conv2 = StraightEdgeConv(embedding_dim // 8, embedding_dim // 4)
        self.conv3 = StraightEdgeConv(embedding_dim // 8 + embedding_dim // 4, embedding_dim)

    def execute(self, x: jt.Var) -> jt.Var:
        idx = knn(x, self.k + 1)[:, :, 1:]
        x1 = self.conv1(x, idx)
        idx = knn(x1, self.k + 1)[:, :, 1:]
        x2 = self.conv2(x1, idx)
        x12 = jt.concat([x1, x2], dim=-1)
        idx = knn(x2, self.k + 1)[:, :, 1:]
        return self.conv3(x12, idx)
