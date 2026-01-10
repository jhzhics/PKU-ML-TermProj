# Kissing Number Solver (Python Framework)

This is a self-contained Python framework to **search for large kissing configurations** (spherical codes)
in **d dimensions** using **evolutionary search + local refinement**.

## Problem (scaled form used in this repo)

We use the scaled form consistent with your project notes:

Find the **largest n** such that there exist points `x_1, ..., x_n` in the unit ball `||x_i|| <= 1` with
pairwise separation:

- `||x_i - x_j|| >= 1` for all `i != j`.

Classic kissing-number (unit spheres kissing a central unit sphere) is equivalent up to scaling: multiply
all coordinates by 2 and the thresholds become `||x_i|| = 2` and `||x_i-x_j|| >= 2`.

## Scoring (matches the project note)

- If constraints are violated: **score = 0**
- If constraints are satisfied: **score = n + empty_radius**

`empty_radius` is defined as the **largest radius r in [0,1]** such that there exists a point `y` with:
- `||y|| = r`
- `||y - x_i|| >= 1` for all i

So `empty_radius >= 1` means you can place an additional point on the boundary and likely increase `n`
(we flag this in evaluation output).

## Quickstart

```bash
pip install -r requirements.txt
python -m experiments.run_search --d 3 --start-n 6 --max-n 20 --budget 4000 --restarts 8 --out out_d3
```

Evaluate a saved solution:

```bash
python -m experiments.evaluate_solution --path out_d3/best_n12.json --rational
```

## Structure

- `src/kissing/*` : geometry, constraints, metrics, initializers, IO
- `src/optim/*`   : evolutionary search + local refinement + OpenEvolve bridge stub
- `experiments/*` : runnable entry points

## OpenEvolve integration

Your prompt referenced OpenEvolve. Since this environment doesn’t include its code, this repo provides:
- a clean **evaluation API** (`evaluate_candidate_points`)
- a **bridge stub** (`src/optim/openevolve_bridge.py`)

Once you clone/install OpenEvolve, adapt that file to match its actual API.

