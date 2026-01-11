from __future__ import annotations
import numpy as np
from .geometry import pairwise_distances

def check_constraints(
    x: np.ndarray,
    min_dist: float = 1.0,
    max_norm: float = 1.0,
    tol: float = 1e-8,
) -> tuple[bool, dict]:
    """
    Constraints:
      (1) ||x_i|| <= max_norm
      (2) ||x_i - x_j|| >= min_dist for all i != j
    Returns: (feasible, info)
    """
    norms = np.linalg.norm(x, axis=1)
    max_norm_violation = float(np.max(norms - max_norm))

    dist = pairwise_distances(x)
    n = x.shape[0]
    dist = dist + np.eye(n) * 1e9
    min_pair = float(np.min(dist))
    min_dist_violation = float(min_dist - min_pair)

    feasible = (max_norm_violation <= tol) and (min_dist_violation <= tol)
    info = {
        "max_norm": float(np.max(norms)),
        "min_pair_dist": min_pair,
        "max_norm_violation": max_norm_violation,
        "min_dist_violation": min_dist_violation,
        "min_dist": float(min_dist),
        "max_norm_allowed": float(max_norm),
        "tol": float(tol),
    }
    return feasible, info

def constraint_violations(
    x: np.ndarray,
    min_dist: float = 1.0,
    max_norm: float = 1.0,
) -> dict:
    """Continuous violation magnitudes (>=0) for penalties."""
    norms = np.linalg.norm(x, axis=1)
    v_norm = np.maximum(0.0, norms - max_norm)

    dist = pairwise_distances(x)
    n = x.shape[0]
    dist = dist + np.eye(n) * 1e9
    v_pair = np.maximum(0.0, min_dist - dist)
    v_pair = v_pair[np.triu_indices(n, k=1)]
    return {
        "norm_violation_sum": float(np.sum(v_norm)),
        "pair_violation_sum": float(np.sum(v_pair)),
        "norm_violation_max": float(np.max(v_norm)) if len(v_norm) else 0.0,
        "pair_violation_max": float(np.max(v_pair)) if len(v_pair) else 0.0,
    }
