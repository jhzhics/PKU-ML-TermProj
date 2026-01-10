from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from tqdm import trange

from src.kissing.constraints import check_constraints, constraint_violations
from src.kissing.metrics import score_configuration
from src.kissing.geometry import random_points_in_unit_ball

@dataclass
class EvolutionConfig:
    population: int = 128
    elite_frac: float = 0.15
    mutation_sigma: float = 0.08
    mutation_rate: float = 0.9
    crossover_rate: float = 0.15
    iters: int = 4000
    local_refine_every: int = 40
    refine_steps: int = 120
    refine_lr: float = 0.03
    seed: int = 0

class EvolutionarySearch:
    """
    Simple evolutionary search for a feasible configuration with high score.

    Representation: each individual is an (n,d) array inside the unit ball.
    """
    def __init__(
        self,
        n: int,
        d: int,
        *,
        min_dist: float = 1.0,
        max_norm: float = 1.0,
        cfg: EvolutionConfig | None = None,
        init_points: np.ndarray | None = None,
    ):
        self.n = int(n)
        self.d = int(d)
        self.min_dist = float(min_dist)
        self.max_norm = float(max_norm)
        self.cfg = cfg or EvolutionConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

        pop = []
        if init_points is not None:
            init = init_points.astype(np.float64)
            assert init.shape == (self.n, self.d)
            pop.append(init)
        while len(pop) < self.cfg.population:
            pop.append(random_points_in_unit_ball(self.rng, self.n, self.d))
        self.pop = np.stack(pop, axis=0)  # (P,n,d)

    def _shaped_fitness(self, x: np.ndarray) -> float:
        """
        Internal fitness used for selection.
        True score is only used when feasible; otherwise we use a negative penalty.
        """
        feasible, _ = check_constraints(x, min_dist=self.min_dist, max_norm=self.max_norm, tol=1e-8)
        if feasible:
            res = score_configuration(
                x, min_dist=self.min_dist, max_norm=self.max_norm,
                empty_radius_directions=2048, empty_radius_seed=0
            )
            return float(res.score)
        vio = constraint_violations(x, min_dist=self.min_dist, max_norm=self.max_norm)
        penalty = 50.0 * vio["pair_violation_sum"] + 20.0 * vio["norm_violation_sum"]
        return float(-penalty)

    def _mutate(self, x: np.ndarray) -> np.ndarray:
        y = x.copy()
        if self.rng.random() < self.cfg.mutation_rate:
            y += self.rng.normal(scale=self.cfg.mutation_sigma, size=y.shape)
        norms = np.linalg.norm(y, axis=1, keepdims=True) + 1e-12
        scale = np.minimum(1.0, self.max_norm / norms)
        return y * scale

    def _crossover(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if self.rng.random() > self.cfg.crossover_rate:
            return a.copy()
        mask = self.rng.random(size=(self.n, 1)) < 0.5
        return np.where(mask, a, b)

    def run(self, *, verbose: bool = True) -> dict:
        from src.optim.local_search import local_refine

        P = self.cfg.population
        elite = max(1, int(self.cfg.elite_frac * P))

        best_x = None
        best_fit = -1e18

        iterator = trange(self.cfg.iters, disable=not verbose, desc=f"EvoSearch n={self.n}, d={self.d}")
        for t in iterator:
            fits = np.array([self._shaped_fitness(self.pop[i]) for i in range(P)], dtype=np.float64)

            idx = int(np.argmax(fits))
            if fits[idx] > best_fit:
                best_fit = float(fits[idx])
                best_x = self.pop[idx].copy()

            if verbose and (t % 50 == 0):
                iterator.set_postfix(best=float(best_fit))

            elite_idx = np.argsort(-fits)[:elite]
            elites = self.pop[elite_idx]

            next_pop = []
            next_pop.extend(list(elites))
            while len(next_pop) < P:
                p1 = elites[self.rng.integers(0, elite)]
                p2 = elites[self.rng.integers(0, elite)]
                child = self._crossover(p1, p2)
                child = self._mutate(child)
                next_pop.append(child)
            self.pop = np.stack(next_pop, axis=0)

            if self.cfg.local_refine_every > 0 and (t + 1) % self.cfg.local_refine_every == 0:
                for i in range(min(4, elite)):
                    self.pop[i] = local_refine(
                        self.pop[i],
                        min_dist=self.min_dist,
                        max_norm=self.max_norm,
                        steps=self.cfg.refine_steps,
                        lr=self.cfg.refine_lr,
                        seed=int(self.cfg.seed + t + i),
                    )

        assert best_x is not None
        res = score_configuration(best_x, min_dist=self.min_dist, max_norm=self.max_norm, empty_radius_directions=8192, empty_radius_seed=0)
        return {"points": best_x, "best_shaped_fitness": best_fit, "evaluation": res}
