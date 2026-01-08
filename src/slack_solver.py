import torch
import torch.optim as optim
import torch.nn.init as init
import numpy as np
import validator

def create_regular_simplex(n):
    """Construct a regular simplex in (n-1)-dimensional space."""
    V = np.eye(n)
    V_centered = V - np.mean(V, axis=0)
    norms = np.linalg.norm(V_centered, axis=1, keepdims=True)
    V_unit = V_centered / norms
    
    u, s, vh = np.linalg.svd(V_unit)
    V_reduced = u[:, :n-1] @ np.diag(s[:n-1])
    
    V_reduced /= np.linalg.norm(V_reduced, axis=1, keepdims=True)
    return torch.tensor(V_reduced, dtype=torch.float32)

def direct_compress(n, target_d, total_steps=10000):
    """Directly compress a regular simplex from (n-1)D to target_d dimensions."""
    V_high = create_regular_simplex(n)
    input_dim = n - 1
    P = torch.randn(input_dim, target_d, requires_grad=True)
    init.orthogonal_(P)
    
    optimizer = optim.Adam([P], lr=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    mask = torch.triu(torch.ones(n, n), diagonal=1).bool()

    for step in range(total_steps):
        optimizer.zero_grad()
        
        V_proj = V_high @ P
        V_normed = V_proj / V_proj.norm(dim=1, keepdim=True)
        
        G = V_normed @ V_normed.t()
        off_diag = G[mask]
        
        loss = torch.logsumexp(off_diag * 60, dim=0)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if step % 2000 == 0:
            current_max = off_diag.max().item()
            print(f"Step {step:5d} | Loss: {loss.item():.4f} | Max Inner Product: {current_max:.6f}")

    return V_normed.detach()

def report_results(points, n, d, label):
    """Print a report of the results."""
    G = points @ points.t()
    mask = torch.triu(torch.ones(n, n), diagonal=1).bool()
    rho = G[mask].max().item()
    
    print("-" * 40)
    print(f"[{label}] Dimensions: {d}, Points: {n}")
    print(f"Max Inner Product (rho): {rho:.6f}")
    if rho <= 0.5:
        print("Status: Success")
    else:
        print(f"Status: Failed")

if __name__ == "__main__":
    num_points = 36
    dim = 5
    
    print(f"Task: Place {num_points} points in {dim} dimensions.")

    points_direct = direct_compress(num_points, dim)
    report_results(points_direct, num_points, dim, "Direct Compression")
    final_points_numpy = points_direct.detach().to(torch.float64).numpy()
    if validator.is_accepted_solution(final_points_numpy):
        print("Verification: The solution is accepted by the validator.")
    else:
        print("Verification: The solution is NOT accepted by the validator.")