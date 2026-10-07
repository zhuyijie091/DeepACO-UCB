import math
import torch
from torch.distributions import Categorical

class ACO():

    def __init__(self, 
                 distances,
                 n_ants=20, 
                 decay=0.9,
                 alpha=1,
                 beta=1,
                 elitist=False,
                 min_max=False,
                 pheromone=None,
                 heuristic=None,
                 min=None,
                #  use_ucb: bool = False,
                 ucb_c: float = 0.0,
                 ucb_schedule=None,
                 ucb_c_min: float = 0.0,
                 ucb_tau=None,
                 stagnation_k: int = 0,
                 boost_gamma: float = 2.0,
                 ucb_c_max=None,
                 boost_cooldown: int = 0,
                 debug: bool = False,
                 debug_every: int = 1,
                 device='cpu'
                 ):
        
        self.problem_size = len(distances)
        self.distances  = distances
        self.n_ants = n_ants
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.elitist = elitist
        self.min_max = min_max
        self.device = device

        # self.use_ucb = use_ucb
        self.ucb_c = float(ucb_c)
        self.ucb_schedule = ucb_schedule
        self.ucb_c_min = float(ucb_c_min)
        self.ucb_tau = None if ucb_tau is None else float(ucb_tau)
        self.stagnation_k = int(stagnation_k)
        self.boost_gamma = float(boost_gamma)
        self.ucb_c_max = None if ucb_c_max is None else float(ucb_c_max)
        self.boost_cooldown = int(boost_cooldown)
        self.debug = bool(debug)
        self.debug_every = int(debug_every)
        self.debug_log = []

        self.iter = 0
        self._no_improve = 0
        self._boost_cooldown_left = 0
        self._ucb_c_current = self.ucb_c
        
        if min_max:
            if min is not None:
                assert min > 1e-9
            else:
                min = 0.1
            self.min = min
            self.max = None
        
        if pheromone is None:
            self.pheromone = torch.ones_like(self.distances)
            if min_max:
                self.pheromone = self.pheromone * self.min
        else:
            self.pheromone = pheromone

        self.heuristic = 1 / distances if heuristic is None else heuristic

        # UCB/UCT exploration statistics (edge-level, within one TSP instance).
        self.edge_visits = torch.zeros_like(self.distances, dtype=torch.float, device=self.device)

        self.shortest_path = None
        self.lowest_cost = float('inf')

    def _maybe_debug_snapshot(self):
        if not self.debug:
            return
        if self.debug_every <= 0:
            return
        if (self.iter % self.debug_every) != 0:
            return

        # Compute global bonus stats from visit counts.
        parent_visits = self.edge_visits.sum(dim=-1, keepdim=True)
        bonus_mat = self._ucb_c_current * torch.sqrt(torch.log(parent_visits + 2.0) / (self.edge_visits + 1.0))

        phe = self.pheromone
        bonus_over_phe = bonus_mat / (phe + 1e-12)

        self.debug_log.append({
            'iter': int(self.iter),
            'ucb_c_current': float(self._ucb_c_current),
            'lowest_cost': float(self.lowest_cost),
            'no_improve': int(self._no_improve),
            'pheromone_min': float(phe.min()),
            'pheromone_mean': float(phe.mean()),
            'pheromone_max': float(phe.max()),
            'bonus_min': float(bonus_mat.min()),
            'bonus_mean': float(bonus_mat.mean()),
            'bonus_max': float(bonus_mat.max()),
            'bonus_over_pheromone_mean': float(bonus_over_phe.mean()),
            'edge_visit_coverage': float((self.edge_visits > 0).float().mean()),
            'edge_visits_max': float(self.edge_visits.max()),
        })

    def _update_ucb_c_current(self):
        c = self.ucb_c

        # Optional exponential decay schedule from ucb_c to ucb_c_min.
        if self.ucb_schedule is not None:
            if self.ucb_schedule != 'exp':
                raise ValueError(f"Unknown ucb_schedule={self.ucb_schedule!r}, expected 'exp' or None")
            if self.ucb_tau is None or self.ucb_tau <= 0:
                raise ValueError("ucb_tau must be a positive float when ucb_schedule='exp'")
            c = self.ucb_c_min + (self.ucb_c - self.ucb_c_min) * math.exp(-self.iter / self.ucb_tau)

        # Optional stagnation-triggered boost.
        if self.stagnation_k > 0 and self.boost_gamma > 1.0:
            if self._boost_cooldown_left > 0:
                self._boost_cooldown_left -= 1
            elif self._no_improve >= self.stagnation_k:
                c = c * self.boost_gamma
                if self.ucb_c_max is not None:
                    c = min(c, self.ucb_c_max)
                self._no_improve = 0
                self._boost_cooldown_left = self.boost_cooldown

        self._ucb_c_current = float(c)



    @torch.no_grad()
    def sparsify(self, k_sparse):
        '''
        Sparsify the TSP graph to obtain the heuristic information 
        Used for vanilla ACO baselines
        '''
        _, topk_indices = torch.topk(self.distances, 
                                        k=k_sparse, 
                                        dim=1, largest=False)
        edge_index_u = torch.repeat_interleave(
            torch.arange(len(self.distances), device=self.device),
            repeats=k_sparse
            )
        edge_index_v = torch.flatten(topk_indices)
        sparse_distances = torch.ones_like(self.distances) * 1e10
        sparse_distances[edge_index_u, edge_index_v] = self.distances[edge_index_u, edge_index_v]
        self.heuristic = 1 / sparse_distances
    
    def sample(self):
        paths, log_probs = self.gen_path(require_prob=True)
        costs = self.gen_path_costs(paths)
        return costs, log_probs

    @torch.no_grad()
    def run(self, n_iterations):
        for _ in range(n_iterations):
            self._update_ucb_c_current()
            paths = self.gen_path(require_prob=False)
            costs = self.gen_path_costs(paths)
            
            best_cost, best_idx = costs.min(dim=0)
            if best_cost < self.lowest_cost:
                self.shortest_path = paths[:, best_idx]
                self.lowest_cost = best_cost
                self._no_improve = 0
                if self.min_max:
                    max = self.problem_size / self.lowest_cost
                    if self.max is None:
                        self.pheromone *= max/self.pheromone.max()
                    self.max = max
            else:
                self._no_improve += 1
            
            self.update_pheronome(paths, costs)
            self._maybe_debug_snapshot()
            self.iter += 1

        return self.lowest_cost
       
    @torch.no_grad()
    def update_pheronome(self, paths, costs):
        '''
        Args:
            paths: torch tensor with shape (problem_size, n_ants)
            costs: torch tensor with shape (n_ants,)
        '''
        self.pheromone = self.pheromone * self.decay 
        
        if self.elitist:
            best_cost, best_idx = costs.min(dim=0)
            best_tour= paths[:, best_idx]
            self.pheromone[best_tour, torch.roll(best_tour, shifts=1)] += 1.0/best_cost
            self.pheromone[torch.roll(best_tour, shifts=1), best_tour] += 1.0/best_cost
        
        else:
            for i in range(self.n_ants):
                path = paths[:, i]
                cost = costs[i]
                self.pheromone[path, torch.roll(path, shifts=1)] += 1.0/cost
                self.pheromone[torch.roll(path, shifts=1), path] += 1.0/cost
                
        if self.min_max:
            self.pheromone[(self.pheromone > 1e-9) * (self.pheromone) < self.min] = self.min
            self.pheromone[self.pheromone > self.max] = self.max
    
    @torch.no_grad()
    def gen_path_costs(self, paths):
        '''
        Args:
            paths: torch tensor with shape (problem_size, n_ants)
        Returns:
                Lengths of paths: torch tensor with shape (n_ants,)
        '''
        assert paths.shape == (self.problem_size, self.n_ants)
        u = paths.T # shape: (n_ants, problem_size)
        v = torch.roll(u, shifts=1, dims=1)  # shape: (n_ants, problem_size)
        assert (self.distances[u, v] > 0).all()
        return torch.sum(self.distances[u, v], dim=1)

    def gen_path(self, require_prob=False):
        '''
        Tour contruction for all ants
        Returns:
            paths: torch tensor with shape (problem_size, n_ants), paths[:, i] is the constructed tour of the ith ant
            log_probs: torch tensor with shape (problem_size, n_ants), log_probs[i, j] is the log_prob of the ith action of the jth ant
        '''
        start = torch.randint(low=0, high=self.problem_size, size=(self.n_ants,), device=self.device)
        mask = torch.ones(size=(self.n_ants, self.problem_size), device=self.device)
        mask[torch.arange(self.n_ants, device=self.device), start] = 0
        
        paths_list = [] # paths_list[i] is the ith move (tensor) for all ants
        paths_list.append(start)
        
        log_probs_list = [] # log_probs_list[i] is the ith log_prob (tensor) for all ants' actions
        
        prev = start
        for _ in range(self.problem_size-1):
            actions, log_probs = self.pick_move(prev, mask, require_prob)
            paths_list.append(actions)
            if require_prob:
                log_probs_list.append(log_probs)
                mask = mask.clone()
            prev = actions
            mask[torch.arange(self.n_ants, device=self.device), actions] = 0
        
        if require_prob:
            return torch.stack(paths_list), torch.stack(log_probs_list)
        else:
            return torch.stack(paths_list)
        
    def pick_move(self, prev, mask, require_prob):
        '''
        Args:
            prev: tensor with shape (n_ants,), previous nodes for all ants
            mask: bool tensor with shape (n_ants, p_size), masks (0) for the visited cities
        '''
        pheromone = self.pheromone[prev] # shape: (n_ants, p_size)
        heuristic = self.heuristic[prev] # shape: (n_ants, p_size)
        visits = self.edge_visits[prev]  # shape: (n_ants, p_size)
        parent_visits = visits.sum(dim=-1, keepdim=True)  # shape: (n_ants, 1)

        heuristic_term = heuristic ** self.beta  # shape: (n_ants, p_size)

        # UCT-style bonus:
        # bonus_ij = c * sqrt(log(N(parent)+1)/(N_ij+1))
        bonus = self._ucb_c_current * torch.sqrt(
            torch.log(parent_visits + 2.0) / (visits + 1.0)
        )

        # Treat (pheromone + bonus) as the effective pheromone matrix.
        pheromone_plus = pheromone + bonus
        pheromone_term = pheromone_plus  ** self.alpha  # shape: (n_ants, p_size)

        dist = pheromone_term * heuristic_term * mask

        dist = Categorical(dist)
        actions = dist.sample() # shape: (n_ants,)
        log_probs = dist.log_prob(actions) if require_prob else None # shape: (n_ants,)

        # Update global edge visit counts after sampling.
        self.edge_visits[prev, actions] += 1.0

        return actions, log_probs
        


if __name__ == '__main__':
    torch.set_printoptions(precision=3,sci_mode=False)
    input = torch.rand(size=(5, 2))
    distances = torch.norm(input[:, None] - input, dim=2, p=2)
    distances[torch.arange(len(distances)), torch.arange(len(distances))] = 1e10
    aco = ACO(distances)
    aco.sparsify(k_sparse=3)
    print(aco.run(20))