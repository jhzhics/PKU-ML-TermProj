from __future__ import annotations
import argparse
import numpy as np
from src.kissing.io import load_solution_json, maybe_rationalize
from src.kissing.metrics import score_configuration

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--path", type=str, required=True)
    p.add_argument("--min-dist", type=float, default=1.0)
    p.add_argument("--max-norm", type=float, default=1.0)
    p.add_argument("--empty-directions", type=int, default=8192)
    p.add_argument("--rational", action="store_true")
    args = p.parse_args()

    payload = load_solution_json(args.path)
    pts = np.array(payload["points"], dtype=np.float64)

    res = score_configuration(
        pts,
        min_dist=args.min_dist,
        max_norm=args.max_norm,
        empty_radius_directions=args.empty_directions,
        empty_radius_seed=0,
    )

    print(f"n={res.n}")
    print(f"feasible={res.feasible}")
    print(f"min_pair_dist={res.constraint_info['min_pair_dist']:.8f} (>= {args.min_dist})")
    print(f"max_norm={res.constraint_info['max_norm']:.8f} (<= {args.max_norm})")
    print(f"empty_radius={res.empty_radius:.8f}")
    print(f"WARNING add-one-possible={res.warning_add_one_possible}")
    print(f"score={res.score:.8f}")

    if args.rational:
        rat = maybe_rationalize(pts)
        print("\nRational-ish coordinates (strings):")
        for row in rat:
            print("[" + ", ".join(row) + "]")

if __name__ == "__main__":
    main()
