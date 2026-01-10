import torch
import torch.optim as optim
import torch.nn.init as init
import numpy as np

def solve(batch: int, n: int, d: int, device: torch.device, total_steps: int = 10000, lr: float = 0.01) -> torch.Tensor:
    V_high = create_regular_simplex(n).to(device = device, dtype = torch.float32) # (n, n-1)
    input_dim = n - 1
    
    P = torch.randn(batch, input_dim, d, device = device, requires_grad=True)
    with torch.no_grad():
        for i in range(batch):
            init.orthogonal_(P[i])
    
    optimizer = optim.Adam([P], lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    mask = torch.triu(torch.ones(n, n), diagonal=1).bool()

    for step in range(total_steps):
        optimizer.zero_grad()
        V_proj = torch.matmul(V_high, P) 
        V_normed = V_proj / V_proj.norm(dim=2, keepdim=True).clamp(min=1e-12)
        G = torch.bmm(V_normed, V_normed.transpose(1, 2))
        off_diag = G[:, mask] 
        loss = torch.logsumexp(off_diag * 60, dim=1).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if step % 2000 == 0:
            global_max = off_diag.max().item()
            print(f"Step {step:5d} | Mean Loss: {loss.item():.4f} | Max Inner Product: {global_max:.6f}")

    return V_normed.detach()

def create_regular_simplex(n):
    V = np.eye(n)
    V_centered = V - np.mean(V, axis=0)
    norms = np.linalg.norm(V_centered, axis=1, keepdims=True)
    V_unit = V_centered / norms
    u, s, vh = np.linalg.svd(V_unit)
    V_reduced = u[:, :n-1] @ np.diag(s[:n-1])
    V_reduced /= np.linalg.norm(V_reduced, axis=1, keepdims=True)
    return torch.tensor(V_reduced, dtype=torch.float32)

# 修改 main 函数进行测试
if __name__ == "__main__":
    B = 4
    N = 36
    D = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Task: Generating {B} independent batches of {N} points in {D}D.")
    
    batch_results = solve(B, N, D, device)
    
    import validator
    for i in range(B):
        points = batch_results[i].cpu().to(torch.float64).numpy()
        success = validator.is_accepted_solution(points)
        print(f"Batch {i} | Accepted: {success}")