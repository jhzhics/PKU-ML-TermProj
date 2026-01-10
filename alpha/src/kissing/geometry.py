from __future__ import annotations
import numpy as np

def normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize each row to unit norm (safe)."""
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    nrm = np.maximum(nrm, eps)
    return x / nrm

def random_unit_vectors(rng: np.random.Generator, n: int, d: int) -> np.ndarray:
    """Sample n random unit vectors uniformly (Gaussian -> normalize)."""
    x = rng.normal(size=(n, d))
    return normalize_rows(x)

def pairwise_distances(x: np.ndarray) -> np.ndarray:
    """
    Full pairwise Euclidean distance matrix (n x n) using:
      ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    """
    s = np.sum(x * x, axis=1, keepdims=True)  # (n,1)
    d2 = s + s.T - 2.0 * (x @ x.T)
    d2 = np.maximum(d2, 0.0)
    return np.sqrt(d2)

def min_pairwise_distance(x: np.ndarray) -> float:
    """Minimum distance among all pairs i<j."""
    n = x.shape[0]
    if n <= 1:
        return float("inf")
    dist = pairwise_distances(x)
    dist = dist + np.eye(n) * 1e9
    return float(np.min(dist))

def project_to_unit_ball(x: np.ndarray, max_norm: float = 1.0, eps: float = 1e-12) -> np.ndarray:
    """Project points into ||x|| <= max_norm by radial clipping."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    scale = np.minimum(1.0, max_norm / np.maximum(norms, eps))
    return x * scale

def random_points_in_unit_ball(rng: np.random.Generator, n: int, d: int) -> np.ndarray:
    """
    Approx uniform in unit ball: direction uniform on sphere, radius ~ U(0,1)^(1/d).
    """
    u = random_unit_vectors(rng, n, d)
    r = rng.random(size=(n, 1)) ** (1.0 / d)
    return u * r
