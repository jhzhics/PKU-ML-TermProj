from __future__ import annotations
import argparse, os, json
import numpy as np

from src.kissing.initializers import (
    init_random_points, init_cross_polytope, init_simplex, init_icosahedron_d3, init_24cell_d4
)
from src.kissing.io import save_solution_json
from src.optim.evolutionary import EvolutionarySearch, EvolutionConfig
from src.optim.local_search import local_refine

def pick_initializer(d: int, n: int, rng: np.random.Generator) -> np.ndarray:
    if d == 3 and n == 12:
        return init_icosahedron_d3()
    if d == 4 and n == 24:
        return init_24cell_d4()
    if n == 2 * d:
        return init_cross_polytope(d)
    if n == d + 1:
        return init_simplex(d)
    return init_random_points(rng, n, d, on_sphere=True)

def try_solve_fixed_n(
    d: int,
    n: int,
    *,
    budget: int,
    seed: int,
    restarts: int,
    min_dist: float,
    max_norm: float,
    out_dir: str,
) -> tuple[dict | None, dict]:
    rng = np.random.default_rng(seed)
    best_payload = None
    best_score = -1e18
    attempt_logs = []

    for r in range(restarts):
        init = pick_initializer(d, n, rng)
        init = local_refine(init, min_dist=min_dist, max_norm=max_norm, steps=150, lr=0.03, seed=seed + 1000 * r)

        cfg = EvolutionConfig(
            population=128,
            iters=budget,
            seed=seed + 1337 * r,
            local_refine_every=30,
            refine_steps=120,
            refine_lr=0.03,
        )
        evo = EvolutionarySearch(n=n, d=d, min_dist=min_dist, max_norm=max_norm, cfg=cfg, init_points=init)
        payload = evo.run(verbose=False)

        res = payload["evaluation"]
        score = float(res.score)

        attempt_logs.append({
            "restart": r,
            "score": score,
            "feasible": bool(res.feasible),
            "empty_radius": float(res.empty_radius),
            "min_pair_dist": float(res.constraint_info.get("min_pair_dist", float("nan"))),
            "max_norm": float(res.constraint_info.get("max_norm", float("nan"))),
        })

        if score > best_score:
            best_score = score
            best_payload = payload

    if best_payload is not None:
        save_solution_json(
            os.path.join(out_dir, f"best_n{n}.json"),
            best_payload["points"],
            meta={
                "d": d, "n": n, "seed": seed, "budget": budget, "restarts": restarts,
                "method": "evolutionary+local_refine",
            },
            score_kwargs=dict(min_dist=min_dist, max_norm=max_norm, empty_radius_directions=8192, empty_radius_seed=0),
        )
    return best_payload, {"attempt_logs": attempt_logs, "best_score": best_score}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, required=True)
    p.add_argument("--start-n", type=int, default=None)
    p.add_argument("--max-n", type=int, default=64)
    p.add_argument("--budget", type=int, default=3000)
    p.add_argument("--restarts", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="out")
    p.add_argument("--min-dist", type=float, default=1.0)
    p.add_argument("--max-norm", type=float, default=1.0)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    start_n = args.start_n
    if start_n is None:
        start_n = max(2 * args.d, args.d + 1)

    global_best_n = 0
    history = []

    for n in range(start_n, args.max_n + 1):
        best, info = try_solve_fixed_n(
            d=args.d, n=n, budget=args.budget, seed=args.seed,
            restarts=args.restarts, min_dist=args.min_dist, max_norm=args.max_norm, out_dir=args.out
        )
        history.append({"n": n, **info})
        feasible = False
        score = 0.0
        if best is not None:
            feasible = bool(best["evaluation"].feasible)
            score = float(best["evaluation"].score)

        print(f"[n={n}] feasible={feasible} score={score:.6f}")
        if feasible:
            global_best_n = n
        else:
            break

    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "d": args.d,
            "start_n": start_n,
            "max_n": args.max_n,
            "best_n_found": global_best_n,
            "history": history,
        }, f, ensure_ascii=False, indent=2)

    if global_best_n > 0:
        print(f"\nBest n found: {global_best_n}. See files in {args.out}/ (best_n*.json, best_overall.json).")
        # Copy best_n into best_overall
        src = os.path.join(args.out, f"best_n{global_best_n}.json")
        dst = os.path.join(args.out, "best_overall.json")
        if os.path.exists(src):
            with open(src, "r", encoding="utf-8") as fsrc:
                data = json.load(fsrc)
            with open(dst, "w", encoding="utf-8") as fdst:
                json.dump(data, fdst, ensure_ascii=False, indent=2)
    else:
        print("\nNo feasible configuration found in the attempted range.")

if __name__ == "__main__":
    main()
