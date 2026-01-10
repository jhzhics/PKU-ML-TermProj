from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .constraints import check_constraints
from .geometry import normalize_rows

@dataclass
class ScoreResult:
    feasible: bool
    score: float
    n: int
    empty_radius: float
    warning_add_one_possible: bool
    constraint_info: dict

def _rmax_for_direction(u: np.ndarray, x: np.ndarray, min_dist: float = 1.0, max_r: float = 1.0) -> float:
    """
    For fixed unit direction u, compute maximum r in [0,max_r] s.t. ||r u - x_i|| >= min_dist for all i.
    Uses quadratic inequality per constraint:
        r^2 - 2 (u·x_i) r + (||x_i||^2 - min_dist^2) >= 0
    """
    ux = x @ u                       # (n,)
    xx = np.sum(x * x, axis=1)       # (n,)
    c = xx - (min_dist ** 2)
    disc = ux * ux - c               # (n,)

    sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
    r1 = ux - sqrt_disc
    r2 = ux + sqrt_disc

    violates_at_max = (disc >= 0.0) & (r1 < max_r) & (max_r < r2)
    rmax_i = np.where(violates_at_max, r1, max_r)
    rmax_i = np.clip(rmax_i, 0.0, max_r)
    return float(np.min(rmax_i)) if x.shape[0] > 0 else float(max_r)

def empty_radius(
    x: np.ndarray,
    min_dist: float = 1.0,
    max_norm: float = 1.0,
    *,
    n_directions: int = 8192,
    seed: int | None = None,
) -> float:
    """
    Approximate:
        max_{||u||=1} rmax(u)
    by sampling random directions u.
    """
    rng = np.random.default_rng(seed)
    u = rng.normal(size=(n_directions, x.shape[1]))
    u = normalize_rows(u)
    best = 0.0
    for k in range(n_directions):
        rk = _rmax_for_direction(u[k], x, min_dist=min_dist, max_r=max_norm)
        if rk > best:
            best = rk
            if best >= max_norm - 1e-12:
                return float(max_norm)
    return float(best)

def score_configuration(
    x: np.ndarray,
    *,
    min_dist: float = 1.0,
    max_norm: float = 1.0,
    tol: float = 1e-8,
    empty_radius_directions: int = 8192,
    empty_radius_seed: int | None = None,
) -> ScoreResult:
    """
    Project-defined score:
      infeasible -> 0
      feasible   -> n + empty_radius
    """
    feasible, info = check_constraints(x, min_dist=min_dist, max_norm=max_norm, tol=tol)
    n = int(x.shape[0])
    if not feasible:
        return ScoreResult(False, 0.0, n, 0.0, False, info)

    er = empty_radius(
        x, min_dist=min_dist, max_norm=max_norm,
        n_directions=empty_radius_directions, seed=empty_radius_seed
    )
    warn = bool(er >= max_norm - 1e-10)
    return ScoreResult(True, float(n + er), n, float(er), warn, info)
