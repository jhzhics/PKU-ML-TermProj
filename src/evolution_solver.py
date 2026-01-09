import torch
import math
from typing import Annotated

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def inplace_optimize(tensor: Annotated[torch.Tensor, "shape(..., n, d)"], lr: float = 0.01, stop_eps = 1e-3, max_iters: int = 10000):
    '''
    In-place optimization of a population tensor with repulsion solver.
    
    :param tensor: shape (..., n, d) population tensor
    :type tensor: torch.Tensor
    :param stop_eps: the optimization stops when the change in the tensor is below this threshold
    :type stop_eps: float
    :param max_iters: maximum number of optimization iterations
    :type max_iters: int
    '''
    
    orig_shape = tensor.shape
    n = orig_shape[-2]
    d = orig_shape[-1]
    flat_tensor = tensor.view(-1, n, d)
    pop_size = flat_tensor.shape[0]
    
    device = flat_tensor.device
    s = 2.0
    
    with torch.no_grad():
        flat_tensor /= flat_tensor.norm(dim=2, keepdim=True).clamp(min=1e-12)

    for i in range(max_iters):
        prev_flat = flat_tensor.clone().detach()
        
        flat_tensor.requires_grad_(True)
        
        inner_products = torch.bmm(flat_tensor, flat_tensor.transpose(1, 2))
        
        dist_sq = (2.0 - 2.0 * inner_products).clamp(min=1e-14)

        mask = torch.triu(torch.ones(n, n, device=device), diagonal=1)
        energy = torch.sum(mask.expand(pop_size, n, n) * torch.pow(dist_sq, -s/2))
        
        energy.backward()
        
        with torch.no_grad():
            current_lr = lr * math.cos(math.pi * i / (2 * max_iters))
            
            grad = flat_tensor.grad
            if grad is None:
                continue

            eta = -current_lr * grad
            norm_eta = eta.norm(dim=2, keepdim=True).clamp(min=1e-14)

            new_points = flat_tensor * torch.cos(norm_eta) + (eta / norm_eta) * torch.sin(norm_eta)
            new_points /= new_points.norm(dim=2, keepdim=True)
            
            flat_tensor.copy_(new_points)
            
            flat_tensor.grad.zero_()

        diff = torch.norm(flat_tensor - prev_flat, dim=(1, 2)).mean()
        if diff < stop_eps:
            break
    return tensor

class EvolutionSolver():
    def __init__(self, n: int, d: int, pop_size: int, mutation_rate: float, crossover_rate: float,generations: int):
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
        
    def initialize_population(self):
        self.population = torch.randn(self.pop_size, self.n, self.d, device=device)
        self.population /= torch.norm(self.population, dim=2, keepdim=True)
        inplace_optimize(self.population)
        
    @staticmethod
    def calculate_socre(X: torch.Tensor) -> torch.Tensor:
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

        new_points = torch.randn(num_instances, d, device=device)
        new_points /= torch.norm(new_points, dim=-1, keepdim=True).clamp(min=1e-12)
        

        batch_indices = torch.arange(num_instances, device=device)
        flat_tensor[batch_indices, rand_indices, :] = new_points
        

        inplace_optimize(flat_tensor)
        return flat_tensor.view(orig_shape)
    
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

        random_matrix = torch.randn(batch_size, d, d, device=device)
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
        
        inplace_optimize(offspring)

        return offspring.view(orig_shape)
    
    def solve(self, verbose: bool = False) -> torch.Tensor:
        self.initialize_population()
        
        for gen in range(self.generations):
            if verbose:
                scores = self.calculate_socre(self.population)
                best_score = scores.min().item()
                mean_score = scores.mean().item()
                worst_score = scores.max().item()
                
                print(f"Gen {gen:4d} | Best Cosine Similarity: {best_score:.6f}\
| Mean Cosine Similarity: {mean_score:.6f} | Worst Cosine Similarity: {worst_score:.6f}")
            num_mutations = int(self.pop_size * self.mutation_rate)
            num_crossovers = int(self.pop_size * self.crossover_rate)
            
            mutated = self.mutate(self.population[:num_mutations])
            
            parents1_indices = torch.randint(0, self.pop_size, (num_crossovers,), device=device)
            parents2_indices = torch.randint(0, self.pop_size, (num_crossovers,), device=device)
            parents1 = self.population[parents1_indices]
            parents2 = self.population[parents2_indices]
            offspring = self.crossover(parents1, parents2)
            
            combined = torch.cat([self.population, mutated, offspring], dim=0)
            scores = self.calculate_socre(combined)
            _, top_indices = torch.topk(-scores, self.pop_size)
            self.population = combined[top_indices]
        
        return self.population
    
def main():
    evolver = EvolutionSolver(n=33, d=5, pop_size=50, mutation_rate=0.2, crossover_rate=1.0, generations=100)
    final_population = evolver.solve(verbose=True)
    best_score = EvolutionSolver.calculate_socre(final_population)
    best_index = torch.argmin(best_score)
    print("Best solution found with maximum cosine similarity:", best_score[best_index].item())

if __name__ == "__main__":
    main()
    