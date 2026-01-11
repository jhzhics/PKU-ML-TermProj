from __future__ import annotations
import numpy as np

def local_refine(
    x0: np.ndarray,
    *,
    min_dist: float = 1.0,
    max_norm: float = 1.0,
    steps: int = 400,
    lr: float = 0.03,
    device: str = "cpu",
    seed: int | None = None,
) -> np.ndarray:
    """
    Local refinement to reduce constraint violations and increase separation.
    Uses PyTorch if available (recommended). Falls back to a small numpy heuristic.

    Soft objective:
      - penalize norm > max_norm
      - penalize pairwise dist < min_dist
      - softly encourage points near the boundary
    """
    try:
        import torch
    except Exception:
        return _local_refine_numpy(x0, min_dist=min_dist, max_norm=max_norm, steps=steps, lr=lr, seed=seed)

    torch.manual_seed(0 if seed is None else int(seed))
    x = torch.tensor(x0, dtype=torch.float32, device=device, requires_grad=True)

    def pairwise_dist(x):
        s = (x * x).sum(dim=1, keepdim=True)
        d2 = s + s.t() - 2.0 * (x @ x.t())
        d2 = torch.clamp(d2, min=0.0)
        d = torch.sqrt(d2 + 1e-12)
        d = d + torch.eye(d.shape[0], device=device) * 1e9
        return d

    opt = torch.optim.Adam([x], lr=lr)
    for _ in range(int(steps)):
        opt.zero_grad(set_to_none=True)

        norms = torch.sqrt((x * x).sum(dim=1) + 1e-12)
        v_norm = torch.relu(norms - max_norm)
        loss_norm = (v_norm * v_norm).mean()

        dmat = pairwise_dist(x)
        v_pair = torch.relu(min_dist - dmat)
        loss_pair = (v_pair * v_pair).mean()

        loss_out = (torch.relu(max_norm - norms) ** 2).mean()

        loss = 20.0 * loss_pair + 10.0 * loss_norm + 0.2 * loss_out
        loss.backward()
        opt.step()

        # project back into the ball
        with torch.no_grad():
            norms = torch.sqrt((x * x).sum(dim=1, keepdim=True) + 1e-12)
            scale = torch.clamp(max_norm / norms, max=1.0)
            x[:] = x * scale

    return x.detach().cpu().numpy().astype(np.float64)

def _local_refine_numpy(
    x0: np.ndarray,
    *,
    min_dist: float,
    max_norm: float,
    steps: int,
    lr: float,
    seed: int | None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = x0.astype(np.float64).copy()
    n, _ = x.shape
    eps = 1e-12
    for _ in range(int(steps)):
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        if i == j:
            continue
        diff = x[i] - x[j]
        dist = float(np.linalg.norm(diff) + eps)
        if dist < min_dist:
            grad = diff / dist
            x[i] += lr * (min_dist - dist) * grad
            x[j] -= lr * (min_dist - dist) * grad

        norms = np.linalg.norm(x, axis=1, keepdims=True) + eps
        scale = np.minimum(1.0, max_norm / norms)
        x *= scale
    return x
