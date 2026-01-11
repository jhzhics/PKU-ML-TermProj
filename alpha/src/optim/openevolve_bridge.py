from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from src.kissing.metrics import score_configuration

@dataclass
class OpenEvolveBridgeConfig:
    """
    Bridge stub to plug this repo into OpenEvolve.

    Once you clone/install OpenEvolve, adapt this file to match its actual API:
      - OpenEvolve proposes candidates (code/params)
      - Your wrapper turns that into points (n,d)
      - Call evaluate_candidate_points for scoring
    """
    min_dist: float = 1.0
    max_norm: float = 1.0
    empty_radius_directions: int = 4096
    tol: float = 1e-8

def evaluate_candidate_points(points: np.ndarray, cfg: OpenEvolveBridgeConfig) -> dict:
    res = score_configuration(
        points,
        min_dist=cfg.min_dist,
        max_norm=cfg.max_norm,
        tol=cfg.tol,
        empty_radius_directions=cfg.empty_radius_directions,
        empty_radius_seed=0,
    )
    return {
        "score": float(res.score),
        "feasible": bool(res.feasible),
        "n": int(res.n),
        "empty_radius": float(res.empty_radius),
        "warning_add_one_possible": bool(res.warning_add_one_possible),
        "constraint_info": res.constraint_info,
    }

def run_with_openevolve_if_available(*args: Any, **kwargs: Any) -> None:
    try:
        import openevolve  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "OpenEvolve is not installed/importable here. Clone/install it and then adapt this bridge file."
        ) from e
    raise NotImplementedError(
        "OpenEvolve detected but integration is not implemented. Open this file and connect to OpenEvolve entry points."
    )
