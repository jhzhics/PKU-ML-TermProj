from __future__ import annotations
import numpy as np
from .geometry import random_unit_vectors, random_points_in_unit_ball, normalize_rows

def init_random_points(rng: np.random.Generator, n: int, d: int, on_sphere: bool = True) -> np.ndarray:
    """Random init on the unit sphere (default) or in the unit ball."""
    if on_sphere:
        return random_unit_vectors(rng, n, d)
    return random_points_in_unit_ball(rng, n, d)

def init_cross_polytope(d: int) -> np.ndarray:
    """
    Cross-polytope vertices: +/- e_i, i=1..d. (n=2d)
    Distances between distinct vertices are sqrt(2) or 2.
    """
    x = np.vstack([np.eye(d), -np.eye(d)])
    return x.astype(np.float64)

def init_simplex(d: int) -> np.ndarray:
    """
    Regular simplex with n=d+1 points in R^d, normalized to unit sphere.
    Construction: start in R^(d+1), subtract centroid, then drop one coord.
    """
    n = d + 1
    E = np.eye(n)
    centroid = np.mean(E, axis=0, keepdims=True)
    X = E - centroid
    X = X[:, :d]
    return normalize_rows(X).astype(np.float64)

def init_icosahedron_d3() -> np.ndarray:
    """12 vertices of a regular icosahedron in R^3, normalized to unit sphere."""
    phi = (1.0 + 5.0 ** 0.5) / 2.0
    pts = []
    for s1 in (-1.0, 1.0):
        for s2 in (-1.0, 1.0):
            pts.append([0.0, s1, s2 * phi])
            pts.append([s1, s2 * phi, 0.0])
            pts.append([s1 * phi, 0.0, s2])
    X = np.array(pts, dtype=np.float64)
    return normalize_rows(X)

def init_24cell_d4() -> np.ndarray:
    """
    24-cell vertices in R^4:
      all permutations of (±1, ±1, 0, 0) / sqrt(2)
    Achieves kissing number n=24 in 4D.
    """
    from itertools import permutations, product
    pts = []
    base = [1.0, 1.0, 0.0, 0.0]
    for perm in set(permutations(base, 4)):
        perm = list(perm)
        nz = [i for i, v in enumerate(perm) if abs(v) > 1e-12]
        for signs in product([-1.0, 1.0], repeat=len(nz)):
            v = np.array(perm, dtype=np.float64)
            for idx, s in zip(nz, signs):
                v[idx] *= s
            pts.append(v)
    X = np.unique(np.array(pts, dtype=np.float64), axis=0)
    X = X / np.sqrt(2.0)
    return X.astype(np.float64)
