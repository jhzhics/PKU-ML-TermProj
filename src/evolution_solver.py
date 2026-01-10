import torch
import math
from AALM import init_symmetric_points
from slack_solver import solve as slack_solve
import validator
import json
import signal
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64

def optimize(batch_tensor: torch.Tensor) -> torch.Tensor:
    orig_shape = batch_tensor.shape
    n, d = orig_shape[-2], orig_shape[-1]
    x = batch_tensor.view(-1, n, d).detach().clone().requires_grad_(True)
    B = x.shape[0]

    # Initialize multipliers and penalty parameters for each sample in the batch
    lambda_eq = torch.zeros(B, n, device=x.device)
    mu_ineq = torch.zeros(B, n, n, device=x.device)
    rho = torch.ones(B, 1, 1, device=x.device) * 1.0
    
    optimizer = torch.optim.Adam([x], lr=0.01)

    for alm_iter in range(10):
        for inner in range(50):
            optimizer.zero_grad()

            dists = torch.cdist(x, x)
            
            eq_res = torch.norm(x, dim=-1) - 1.0
            ineq_res = torch.clamp(1.0 - dists, min=0.0)
            
            loss_eq = (lambda_eq * eq_res + 0.5 * rho.squeeze(-1) * eq_res**2).sum()
            loss_ineq = (mu_ineq * ineq_res + 0.5 * rho * ineq_res**2).sum()
            
            total_loss = loss_eq + loss_ineq
            total_loss.backward()
            optimizer.step()
            
        with torch.no_grad():
            lambda_eq += rho.squeeze(-1) * eq_res
            mu_ineq = torch.clamp(mu_ineq + rho * ineq_res, min=0.0)
            
            violation = eq_res.abs().max(dim=-1)[0]
            rho[violation > 1e-4] *= 1.5

    return x.view(orig_shape)

class EvolutionSolver():
    def __init__(self, n: int, d: int, pop_size: int, mutation_rate: float, crossover_rate: float,generations: int, early_stop_cost: float = 0.5):
        '''
        :param crossover_rate: Will have popsize * crossover_rate offspring created by crossover
        :type crossover_rate: float
        '''
        self.n = n
        self.d = d
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generations = generations
        self.early_stop_cost = early_stop_cost
    def initialize_population(self):
        """
        Initialize the population using a mix of strategies:
        1. Symmetric Icosahedron
        2. Simplex Projection
        3. Random Exploration
        """
        self.population = torch.zeros(self.pop_size, self.n, self.d, device=device,dtype=dtype)
        
        size_simplex = int(self.pop_size * 0.34)
        size_symmetric = int(self.pop_size * 0.34)
        size_random = self.pop_size - size_simplex - size_symmetric

        # --- First part: Simplex Projection ---
        print(f"Initializing {size_simplex} individuals using Simplex Projection.")
        if size_simplex > 0:
            simplex_inits = slack_solve(
                batch=size_simplex, n=self.n, d=self.d, 
                device=device, total_steps=10000 
            )
            self.population[:size_simplex] = simplex_inits

        # --- Second part: Symmetric Icosahedron ---
        print(f"Initializing {size_symmetric} individuals using Symmetric Icosahedron.")
        idx_start = size_simplex
        idx_end = size_simplex + size_symmetric
        for i in range(idx_start, idx_end):
            self.population[i] = init_symmetric_points(self.d, self.n, device)

        # --- Third part: Random Exploration ---
        print(f"Initializing {size_random} individuals using Random Exploration.")
        if size_random > 0:
            random_pop = torch.randn(size_random, self.n, self.d, device=device)
            random_pop /= random_pop.norm(dim=2, keepdim=True).clamp(min=1e-12)
            self.population[idx_end:] = random_pop

        self.population = optimize(self.population)

        
    @staticmethod
    def calculate_score(X: torch.Tensor) -> torch.Tensor:
        '''
        Calculate the score (maximum cosine similarity) of the given tensor X.
        
        :param X: A tensor of shape (...., n, d)
        :type X: torch.Tensor
        :return: A tensor of shape (...) containing the maximum cosine similarity for each set of points.
        :rtype: Tensor
        '''
        X = X / torch.norm(X, dim=-1, keepdim=True).clamp(min=1e-12)
        inner_products = torch.matmul(X, X.transpose(-1, -2))
        
        n = X.shape[-2]
        device = X.device
        mask = torch.triu(torch.ones(n, n, device=device), diagonal=1).bool()
        fill_value = -2.0 
        masked_inner = inner_products.masked_fill(~mask, fill_value)
        max_cosine, _ = torch.max(masked_inner.flatten(start_dim=-2), dim=-1)
        
        return max_cosine
    
    @staticmethod
    def mutate(tensor: torch.Tensor) -> torch.Tensor:
        '''
        Mutate the given tensor by removing one point and adding a new random point on the unit sphere.
        The returned tensor would have been optimized in-place.
        
        :param tensor: Tensor of shape (..., n, d)
        :type tensor: torch.Tensor
        :return: Mutated tensor of shape (..., n, d)
        :rtype: Tensor
        '''
        orig_shape = tensor.shape
        n = orig_shape[-2]
        d = orig_shape[-1]
        device = tensor.device
        

        flat_tensor = tensor.clone().view(-1, n, d)
        num_instances = flat_tensor.shape[0]
        
        rand_indices = torch.randint(0, n, size=(num_instances,), device=device)

        new_points = torch.randn(num_instances, d, device=device, dtype=dtype)
        new_points /= torch.norm(new_points, dim=-1, keepdim=True).clamp(min=1e-12)
        

        batch_indices = torch.arange(num_instances, device=device)
        flat_tensor[batch_indices, rand_indices, :] = new_points
        

        flat_tensor = optimize(flat_tensor)
        return flat_tensor.view(orig_shape)
    
    def get_crossover_indices(self, num_crossovers: int) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        Get indices for parent selection during crossover.
        This method selects parents such that elite pairs are prioritized.
        '''
        target_elite_count = num_crossovers // 2
        K = int(math.sqrt(target_elite_count))
        K = max(1, min(K, self.pop_size)) 


        idx_range = torch.arange(K, device=device)
        idx_i, idx_j = torch.meshgrid(idx_range, idx_range, indexing='ij')
        
        p1_elite = idx_i.reshape(-1)
        p2_elite = idx_j.reshape(-1)
        
        current_elite_num = p1_elite.shape[0]
        remaining = num_crossovers - current_elite_num
        
        if remaining > 0:
            p1_rand = torch.randint(0, self.pop_size, (remaining,), device=device)
            p2_rand = torch.randint(0, self.pop_size, (remaining,), device=device)
            
            idx1 = torch.cat([p1_elite, p1_rand], dim=0)
            idx2 = torch.cat([p2_elite, p2_rand], dim=0)
        else:
            idx1 = p1_elite[:num_crossovers]
            idx2 = p2_elite[:num_crossovers]
            
        return idx1, idx2
    
    @staticmethod
    def crossover(parent1: torch.Tensor, parent2: torch.Tensor) -> torch.Tensor:
        '''
        Perform crossover between two parent tensors to produce an offspring tensor.
        The method is to take half points from each parent after random rotation.
        
        :param parent1: Tensor of shape (..., n, d)
        :param parent2: Tensor of shape (..., n, d)
        :return: Offspring tensor of shape (..., n, d)
        '''
        
        orig_shape = parent1.shape
        n = orig_shape[-2]
        d = orig_shape[-1]
        device = parent1.device
        
        p1 = parent1.view(-1, n, d).clone()
        p2 = parent2.view(-1, n, d).clone()
        batch_size = p1.shape[0]

        random_matrix = torch.randn(batch_size, d, d, device=device,dtype=dtype)
        q, r = torch.linalg.qr(random_matrix)
        d_sign = torch.diagonal(r, dim1=-2, dim2=-1).sign().view(batch_size, 1, d)
        q = q * d_sign

        p1_rot = torch.bmm(p1, q)
        p2_rot = torch.bmm(p2, q)

        idx1 = torch.argsort(p1_rot[..., 0], dim=-1)
        idx2 = torch.argsort(p2_rot[..., 0], dim=-1)

        gather_idx1 = idx1.unsqueeze(-1).expand(-1, -1, d)
        gather_idx2 = idx2.unsqueeze(-1).expand(-1, -1, d)
        p1_sorted = torch.gather(p1, 1, gather_idx1)
        p2_sorted = torch.gather(p2, 1, gather_idx2)

        k = math.ceil(n / 2)
        offspring = torch.cat([p1_sorted[:, :k, :], p2_sorted[:, k:, :]], dim=1)
        
        offspring = optimize(offspring)

        return offspring.view(orig_shape)
    
    def solve(self, verbose: bool = False) -> torch.Tensor:
        self.initialize_population()
        
        for gen in range(self.generations):
            if verbose:
                scores = self.calculate_score(self.population)
                best_score = scores.min().item()
                mean_score = scores.mean().item()
                worst_score = scores.max().item()
                
                print(f"Gen {gen:4d} | Best Cosine Similarity: {best_score:.6f}\
| Mean Cosine Similarity: {mean_score:.6f} | Worst Cosine Similarity: {worst_score:.6f}")
                
                if best_score <= self.early_stop_cost:
                    print(f"Early stopping at generation {gen} with best score {best_score:.6f}")
                    break
            num_mutations = int(self.pop_size * self.mutation_rate)
            num_crossovers = int(self.pop_size * self.crossover_rate)
            
            mutated_indices = torch.randint(0, self.pop_size, (num_mutations,), device=device)
            mutated = self.mutate(self.population[mutated_indices])
            
            parents1_indices, parents2_indices = self.get_crossover_indices(num_crossovers)
            parents1 = self.population[parents1_indices]
            parents2 = self.population[parents2_indices]
            offspring = self.crossover(parents1, parents2)
            
            combined = torch.cat([self.population, mutated, offspring], dim=0)
            scores = self.calculate_score(combined)
            _, top_indices = torch.topk(-scores, self.pop_size)
            self.population = combined[top_indices]
        
        return self.population
    
    def get_current_best(self) -> tuple[torch.Tensor, float]:
        scores = self.calculate_score(self.population)
        best_score, best_index = torch.min(scores, dim=0)
        best_layout = self.population[best_index]
        return best_layout, best_score.item()
    

def save_checkpoint(layout, score, d, n, filename="result.json"):
    print(f"\n[Saving] Saving layout to {filename}...")
    save_data = {
        "dimension": d,
        "num_spheres": n,
        "best_score": float(score),
        "layout": layout.cpu().detach().numpy().tolist()
    }
    with open(filename, "w") as f:
        json.dump(save_data, f)
    print("[Saving] Save complete.\n")
    
    
def main():
    def signal_handler(sig, frame):
        print("\nSaving current best solution and exiting...")
        try:
            current_best_layout, current_best_score = evolver.get_current_best()
            save_checkpoint(current_best_layout, current_best_score, d, n)
        except Exception as e:
            print(f"[Error] Failed to save checkpoint: {e}")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        n = 38
        d = 5
        evolver = EvolutionSolver(n=n, d=d, pop_size=100, mutation_rate=1.0, crossover_rate=2.0, generations=20)
        final_population = evolver.solve(verbose=True)
        best_score = EvolutionSolver.calculate_score(final_population)
        best_index = torch.argmin(best_score)
        print("Best solution found with maximum cosine similarity:", best_score[best_index].item())
        best_layout = final_population[best_index]
        if validator.is_accepted_solution(best_layout.cpu().detach().numpy()):
            print("Found valid solution.")
        else:
            print("Solution is NOT valid.")
            
    except Exception as e:
        print(f"[Error] An exception occurred: {e}")
    finally:
        signal_handler(None, None)
    

if __name__ == "__main__":
    main()
    