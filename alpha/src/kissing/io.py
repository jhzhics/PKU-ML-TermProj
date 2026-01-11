from __future__ import annotations
import json, os
from dataclasses import asdict
import numpy as np
from .metrics import score_configuration, ScoreResult

def save_solution_json(path: str, points: np.ndarray, *, meta: dict | None = None, score_kwargs: dict | None = None) -> dict:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    score_kwargs = score_kwargs or {}
    res: ScoreResult = score_configuration(points, **score_kwargs)
    payload = {"points": points.tolist(), "meta": meta or {}, "evaluation": asdict(res)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload

def load_solution_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def maybe_rationalize(points: np.ndarray, max_den: int = 1000, tol: float = 1e-6) -> list[list[str]]:
    """
    Try to express floats as rationals (strings) when possible.
    Uses sympy if available; otherwise prints decimals.
    """
    try:
        import sympy as sp
    except Exception:
        return [[f"{float(v):.8f}" for v in row] for row in points]

    out: list[list[str]] = []
    for row in points:
        rrow: list[str] = []
        for v in row:
            rat = sp.nsimplify(float(v), [], tolerance=tol, rational=True, maxsteps=50)
            if hasattr(rat, "q") and int(rat.q) <= max_den:
                rrow.append(str(rat))
            else:
                rrow.append(f"{float(v):.8f}")
        out.append(rrow)
    return out
