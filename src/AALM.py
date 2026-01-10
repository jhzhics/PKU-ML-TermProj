import numpy as np
import torch
import math
import time
import validator


SPHERE_RADIUS = 2.0      
MIN_PAIR_DISTANCE = 2.0   
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float64     


SMOOTH_BETA = 10.0        
PERTURB_THRESHOLD = 1e-5  
PERTURB_PATIENCE = 5      
PERTURB_MAGNITUDE = 0.005 
RESIDUAL_THRESHOLD = 1e-2 


def project_to_sphere(positions: torch.Tensor) -> torch.Tensor:

    norms = torch.norm(positions, dim=1, keepdim=True)
    positions_proj = positions * (SPHERE_RADIUS / norms.clamp(min=1e-14))
    return positions_proj

def compute_min_pair_distance(positions: torch.Tensor) -> torch.Tensor:

    dist_matrix = torch.cdist(positions, positions, p=2)
    mask = torch.triu(torch.ones_like(dist_matrix), diagonal=1).bool()
    pair_distances = dist_matrix[mask]
    return torch.min(pair_distances) if len(pair_distances) > 0 else torch.tensor(0.0, device=DEVICE)

def check_constraints(positions: torch.Tensor) -> tuple[bool, float, float]:

  
    sphere_errors = torch.abs(torch.norm(positions, dim=1) - SPHERE_RADIUS)
    max_sphere_error = torch.max(sphere_errors).item()
    
  
    min_pair_dist = compute_min_pair_distance(positions).item()
    
    
    is_valid = (max_sphere_error < 1e-8) and (min_pair_dist >= MIN_PAIR_DISTANCE - 1e-8)
    return is_valid, max_sphere_error, min_pair_dist

def init_symmetric_points(dim: int, n: int, device: torch.device, DTYPE = torch.float64) -> torch.Tensor:

    phi = (1 + math.sqrt(5)) / 2  
    base_3d = torch.tensor([
        (0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi),
        (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0),
        (phi, 0, 1), (-phi, 0, 1), (phi, 0, -1), (-phi, 0, -1)
    ], device=device, dtype=DTYPE)
    

    pad = torch.zeros((12, dim - 3), device=device, dtype=DTYPE)
    base_highd = torch.cat([base_3d, pad], dim=1)
    
    positions = []
    while len(positions) < n:
       
        q, _ = torch.qr(torch.randn(dim, dim, device=device, dtype=DTYPE))
        rotated = base_highd @ q.T 
        positions.extend(rotated.unbind(0))
    positions = torch.stack(positions[:n], dim=0)
    

    positions = project_to_sphere(positions)
    positions += torch.randn_like(positions) * 0.01  
    
    return positions


def alm_solve_kissing_number(
    x: torch.tensor,   # (n, d)
    dim: int,         
    n: int,            
    max_alm_iter: int = 100,   
    max_inner_iter: int = 200, 
    initial_rho: float = 1.0,  
    max_rho: float = 1e7,      
    lbfgs_lr: float = 1.0     
) -> tuple[bool, float, torch.Tensor]:

    # x = init_symmetric_points(dim, n, DEVICE)
    # x.requires_grad_(True)
    

    with torch.no_grad():
        
        eq_residual_init = torch.norm(x.data, dim=1) - SPHERE_RADIUS
        lambda_eq = 0.1 * eq_residual_init
        

        dist_matrix_init = torch.cdist(x.data, x.data, p=2)
        ineq_residual_init = MIN_PAIR_DISTANCE - dist_matrix_init
        mu_ineq = 0.01 * torch.clamp(ineq_residual_init, min=0.0)
    

    rho = initial_rho
    best_min_dist = 0.0
    best_x = x.data.clone()
    

    patience_counter = 0
    prev_min_dist = compute_min_pair_distance(x.data).item()
    
   
    start_time = time.time()
    for alm_iter in range(max_alm_iter):
        
        def augmented_lagrangian():
           
            dist_matrix = torch.cdist(x, x, p=2)
            mask = torch.triu(torch.ones_like(dist_matrix), diagonal=1).bool()
            pair_distances = dist_matrix[mask]
            
           
            if len(pair_distances) == 0:
                energy = torch.tensor(0.0, device=DEVICE)
            else:
                smooth_min = - (1 / SMOOTH_BETA) * torch.logsumexp(-SMOOTH_BETA * pair_distances, dim=0)
                energy = -smooth_min  
            
            eq_residual = torch.norm(x, dim=1) - SPHERE_RADIUS
            eq_term = torch.sum(lambda_eq * eq_residual) + 0.5 * rho * torch.sum(eq_residual ** 2)
            
            
            ineq_residual = torch.clamp(MIN_PAIR_DISTANCE - dist_matrix, min=0.0)
            ineq_term = torch.sum(mu_ineq * ineq_residual) + 0.5 * rho * torch.sum(ineq_residual ** 2)
            
            
            total_loss = energy + eq_term + ineq_term
            return total_loss
        
      
        adam_iter = max_inner_iter // 2  
        lbfgs_iter = max_inner_iter - adam_iter  
        
       
        adam_optimizer = torch.optim.Adam([x], lr=0.001)
        lbfgs_optimizer = torch.optim.LBFGS([x], lr=lbfgs_lr, max_iter=lbfgs_iter, line_search_fn="strong_wolfe")
        
        def closure():
            optimizer = adam_optimizer if closure.iter < adam_iter else lbfgs_optimizer
            optimizer.zero_grad()
            loss = augmented_lagrangian()
            loss.backward()
            closure.iter += 1
            return loss
        
        closure.iter = 0  
       
        for _ in range(max_inner_iter):
            if closure.iter < adam_iter:
                adam_optimizer.step(closure)
            else:
                lbfgs_optimizer.step(closure)
        
      
        with torch.no_grad():
            x.data = project_to_sphere(x.data)
        
        
        current_min_dist = compute_min_pair_distance(x.data).item()
        dist_change = abs(current_min_dist - prev_min_dist)
        
        if dist_change < PERTURB_THRESHOLD:
            patience_counter += 1
            if patience_counter >= PERTURB_PATIENCE:
                with torch.no_grad():
                    
                    q, _ = torch.qr(torch.randn(dim, dim, device=DEVICE, dtype=DTYPE))
                    x.data = x.data @ q.T
                    x.data += torch.randn_like(x.data) * PERTURB_MAGNITUDE  
                    x.data = project_to_sphere(x.data) 
                patience_counter = 0 
                
        else:
            patience_counter = 0
        prev_min_dist = current_min_dist
        
      
        with torch.no_grad():
     
            eq_residual = torch.norm(x.data, dim=1) - SPHERE_RADIUS
            lambda_eq = lambda_eq + rho * eq_residual
            
       
            dist_matrix = torch.cdist(x.data, x.data, p=2)
            ineq_residual = MIN_PAIR_DISTANCE - dist_matrix
            mu_ineq = torch.max(mu_ineq + rho * ineq_residual, torch.tensor(0.0, device=DEVICE))
        
   
        max_eq_residual = torch.max(torch.abs(eq_residual)).item()
        max_ineq_residual = torch.max(torch.clamp(ineq_residual, min=0.0)).item()
        max_residual = max(max_eq_residual, max_ineq_residual)
        
    
        rho_growth = 3.0 if max_residual > RESIDUAL_THRESHOLD else 1.2
        
   
        if current_min_dist > best_min_dist:
            best_min_dist = current_min_dist
            best_x = x.data.clone()
        
        is_valid, max_sphere_error, _ = check_constraints(x.data)
        print(f"iter: {alm_iter+1:2d} | rho: {rho:.2e} | min_dist: {current_min_dist:.4f} | max_sphere_error: {max_sphere_error:.6f} ")
        
    
        if not is_valid:
            rho = min(rho * rho_growth, max_rho)
        else:
            break
    
    
    total_time = time.time() - start_time
    print(f"Total time = {total_time}")
    return best_x


def main():
 
    dim, n = 5, 36
    x = init_symmetric_points(dim, n, DEVICE)
    x.requires_grad_(True)
    print(x.shape)
    positions = alm_solve_kissing_number(x, dim, n)
    if validator.is_accepted_solution(positions.cpu().numpy()):
        print(f"Found valid solution for n={n}, d={dim}.")
    else:
        print(f"Solution for n={n}, d={dim} is NOT valid.")
    


if __name__ == "__main__":
    main()