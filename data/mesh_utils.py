"""Mesh loading, sampling, and normalization utilities."""
from __future__ import annotations

import numpy as np

try:
    import trimesh
except ImportError:
    trimesh = None


def compute_mesh_normals(vertices: np.ndarray, faces: np.ndarray):
    face_vectors_1 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    face_vectors_2 = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    face_cross = np.cross(face_vectors_1, face_vectors_2)
    face_norm = np.linalg.norm(face_cross, axis=-1, keepdims=True)
    face_normals = face_cross / np.maximum(face_norm, 1e-12)

    vertex_normals = np.zeros_like(vertices, dtype=np.float64)
    for corner in range(3):
        np.add.at(vertex_normals, faces[:, corner], face_cross)
    vertex_norm = np.linalg.norm(vertex_normals, axis=-1, keepdims=True)
    vertex_normals = vertex_normals / np.maximum(vertex_norm, 1e-12)
    return vertex_normals.astype(np.float32), face_normals.astype(np.float32)


def load_mesh(mesh_path: str):
    if trimesh is None:
        raise ImportError("trimesh is required for mesh loading")
    mesh = trimesh.load(mesh_path, process=False, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    return vertices, faces


def sample_mesh_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    num_samples: int = 32768,
    num_vertex_samples: int = 1024,
):
    """Sample points and normals from mesh surface."""
    if trimesh is not None:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        pts, face_idx = trimesh.sample.sample_surface(mesh, num_samples)
        normals = mesh.face_normals[face_idx].astype(np.float32)
        if num_vertex_samples > 0 and len(vertices) > 0:
            v_idx = np.random.choice(len(vertices), min(num_vertex_samples, len(vertices)), replace=False)
            v_pts = vertices[v_idx].astype(np.float32)
            v_normals, _ = compute_mesh_normals(vertices, faces)
            v_normals = v_normals[v_idx]
            pts = np.concatenate([pts.astype(np.float32), v_pts], axis=0)
            normals = np.concatenate([normals, v_normals], axis=0)
        return pts.astype(np.float32), normals.astype(np.float32)

    # Fallback: sample face centroids
    rng = np.random.default_rng()
    idx = rng.choice(len(faces), size=num_samples, replace=True)
    tri = vertices[faces[idx]]
    u = rng.random((num_samples, 1))
    v = rng.random((num_samples, 1))
    mask = u + v > 1
    u[mask] = 1 - u[mask]
    v[mask] = 1 - v[mask]
    w = 1 - u - v
    pts = (tri[:, 0] * u + tri[:, 1] * v + tri[:, 2] * w).astype(np.float32)
    _, face_normals = compute_mesh_normals(vertices, faces)
    normals = face_normals[idx]
    return pts, normals.astype(np.float32)


def normalize_pc(pc: np.ndarray) -> np.ndarray:
    p_max = pc.max(axis=0)
    p_min = pc.min(axis=0)
    center = (p_max + p_min) / 2.0
    pc = pc - center
    scale = np.sqrt((pc ** 2).sum(axis=1).max())
    scale = max(float(scale), 1e-8)
    return (pc / scale).astype(np.float32)


def sample_points_from_mesh_path(
    mesh_path: str,
    num_samples: int = 32768,
    num_vertex_samples: int = 1024,
    normalize: bool = True,
):
    vertices, faces = load_mesh(mesh_path)
    pts, normals = sample_mesh_surface(vertices, faces, num_samples, num_vertex_samples)
    if normalize:
        pts = normalize_pc(pts)
        # approximate normal preservation under uniform scale about center
    return pts, normals
