import torch
import math
import validator
'''
Dimension KnownLowerBound KnownUpperBound
1 	2
2 	6
3 	12
4 	24
5 	40 	44
6 	72 	77
7 	126 	134
8 	240
9 	306 	363
10 	510 	553
11 	593 	868
12 	840 	1,355
13 	1,154 	2,064
14 	1,932 	3,174
15 	2,564 	4,853
16 	4,320 	7,320
17 	5,730 	10,978
18 	7,654 	16,406
19 	11,692 	24,417
20 	19,448 	36,195
21 	29,768 	53,524
22 	49,896 	80,810
23 	93,150 	122,351
24 	196,560
25 	197,056 	265,006
26 	198,550 	367,775
27 	200,044 	522,212
28 	204,520 	752,292
29 	209,496 	1,075,991
30 	220,440 	1,537,707
31 	238,078 	2,213,487
32 	345,408 	3,162,316 
'''


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    try_solve(32, 5, device)
    
def try_solve(n: int, d: int, device: torch.device):
    X = torch.randn(d, n, device=device, dtype=torch.float64)
    X /= torch.norm(X, dim=0, keepdim=True)
    
    iterations = 10000
    lr = 0.005
    s = 2.0
    min_angle = 0.0
    
    for i in range(iterations):
        X.requires_grad_(True)
        
        inner_products = torch.mm(X.T, X)
        dist_sq = (2.0 - 2.0 * inner_products).clamp(min=1e-14)
        mask = torch.triu(torch.ones(n, n, device=device), diagonal=1)
        energy = torch.sum(mask * torch.pow(dist_sq, -s/2))
        
        energy.backward()
        
        with torch.no_grad():
            current_lr = lr * math.cos(math.pi * i / (2 * iterations))
            eta = -current_lr * X.grad
            norm_eta = torch.norm(eta, dim=0, keepdim=True).clamp(min=1e-14)
            
            X = X * torch.cos(norm_eta) + (eta / norm_eta) * torch.sin(norm_eta)
            X /= torch.norm(X, dim=0, keepdim=True)

        if i % 10 == 0 or i == iterations - 1:
            with torch.no_grad():
                final_inner = torch.mm(X.T, X)
                off_diag = final_inner - torch.eye(n, device=device) * 2.0
                max_cos = torch.max(off_diag)
                min_angle = torch.acos(max_cos.clamp(-1.0, 1.0)) * 180 / math.pi
                print(f"Iter {i:5d} | LR: {current_lr:.4f} | Angle: {min_angle:.4f}°")

    if validator.is_accepted_solution(X.T.cpu().numpy()):
        print(f"Found valid solution for n={n}, d={d} with min angle {min_angle:.4f}°")
    else:
        print(f"Solution for n={n}, d={d} is NOT valid.")
        
if __name__ == "__main__":
    main()
    