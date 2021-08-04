import random
import sys
from timeit import default_timer as timer
import copy
import numpy as np
from bitarray import bitarray

from bloopy.individual import individual
import bloopy.algorithms.hillclimbers as hillclimb
from bloopy.algorithms.local_search import multi_start_local_search_base

class iterative_local_search_base(multi_start_local_search_base):
    def __init__(self, fitness_function, bitstring_size, minmax_problem, random_walk, noimprove=100, searchspace=None, neighbour="Hamming"):
        r"""
        Base Iterative Local Search algorithm.

        Args:
            iterations (int): Number of random restarts.
            fitness_function (bitarray() -> float): Function that
                scores fitness of bitstrings.
            bitstring_size (int): Length of the bitstring instances.
            min_max_problem (int): 1 if maximization problem, -1 for
                    minimization problem. Default is 1.
            random_walk (int): Length of random walk to generate
                new starting candidate.
            noimprove (int): (optional) Terminate if no improvement
                after this many trials. Default is 100.
            searchspace (dict): Mapping of settings to fitnesses
            neighbour (string): Method for generating neighbour solutions to visit.
        """
        super().__init__(fitness_function, bitstring_size, minmax_problem, searchspace=searchspace, neighbour=neighbour)
        self.random_walk = random_walk
        self.noimprove = noimprove
        self.last_improve = 0

    def generate_candidate(self, maxfeval):
        self.randomwalk(maxfeval)
        self.hillclimb_candidate(maxfeval)

        if self.best_candidate is None or self.best_candidate.fitness*self.minmax < self.current_candidate.fitness*self.minmax:
            self.last_improve = 0
            self.best_candidate = self.current_candidate
        else:
            self.last_improve += 1

    def randomwalk(self, maxfeval):
        r"""
        Generate new starting candidate by performing a random walk
            from previous local optimum. Terminate when no improvement
            for certain number of trials.
        """
        if self.best_candidate is None or self.last_improve >= self.noimprove:
            self.last_improve = 0
            self.current_candidate = individual(self.bs_size, boundary_list=self.boundary_list)
            bsstr = self.current_candidate.bitstring.to01()
            if bsstr in self.visited_cache:
                self.current_candidate.fitness = self.visited_cache[bsstr]
            else:
                self.current_candidate.fitness = self.ffunc(self.current_candidate.bitstring)
                self.visited_cache[bsstr] = self.current_candidate.fitness
                self.func_evals += 1
                self.cumulative_fit += self.current_candidate.fitness
        else:
            self.current_candidate = individual(self.best_candidate.size, bitstring=copy.deepcopy(self.best_candidate.bitstring), boundary_list=self.boundary_list)
            self.current_candidate.fitness = self.best_candidate.fitness
            for k in range(self.random_walk):
                self.point_mutate(self.current_candidate)
                
            bsstr = self.current_candidate.bitstring.to01()
            if bsstr in self.visited_cache:
                self.current_candidate.fitness = self.visited_cache[bsstr]
            else:
                self.current_candidate.fitness = self.ffunc(self.current_candidate.bitstring)
                self.visited_cache[bsstr] = self.current_candidate.fitness
                self.func_evals += 1
                self.cumulative_fit += self.current_candidate.fitness

class BestILS(iterative_local_search_base):
    def __init__(self, fitness_function, bitstring_size, minmax_problem, random_walk, noimprove=100, searchspace=None, neighbour='Hamming'):
        r"""
            Best improvement ILS which goes through all neighbours and moves to the best.

        Args:
            iterations (int): Number of random restarts.
            fitness_function (bitarray() -> float): Function that
                scores fitness of bitstrings.
            bitstring_size (int): Length of the bitstring instances.
            min_max_problem (int): 1 if maximization problem, -1 for
                    minimization problem. Default is 1.
            random_walk (int): Length of random walk to generate
                new starting candidate.
            noimprove (int): (optional) Terminate if no improvement
                after this many trials. Default is 100.
            searchspace (dict): Mapping of settings to fitnesses
            neighbour (string): Method for generating neighbour solutions to visit.
        """
        super().__init__(fitness_function, bitstring_size, minmax_problem, random_walk, noimprove, searchspace=searchspace, neighbour=neighbour)

    def hillclimb_candidate(self, maxfeval):
        self.current_candidate, extra_fevals, totfit, self.visited_cache = hillclimb.BestHillclimb(self.current_candidate, self.ffunc, self.minmax, self.func_evals, maxfeval, self.visited_cache, self.nbour_method)
        self.func_evals += extra_fevals
        self.cumulative_fit += totfit

class RandomGreedyILS(iterative_local_search_base):
    def __init__(self, fitness_function, bitstring_size, minmax_problem, random_walk, noimprove=100, searchspace=None, neighbour='Hamming', restart_search=True):
        r"""
            Random greedy ILS which goes through the neighbours at random
             and moves to the first improvement.

        Args:
            iterations (int): Number of random restarts.
            fitness_function (bitarray() -> float): Function that
                scores fitness of bitstrings.
            bitstring_size (int): Length of the bitstring instances.
            min_max_problem (int): 1 if maximization problem, -1 for
                    minimization problem. Default is 1.
            random_walk (int): Length of random walk to generate
                new starting candidate.
            noimprove (int): (optional) Terminate if no improvement
                after this many trials. Default is 100.
            searchspace (dict): Mapping of settings to fitnesses
            neighbour (string): Method for generating neighbour solutions to visit.
            restartsearch (bool): Bool to decide whether we keep searching in the loop,
                    or break out of the loop en hillclimb from the start
        """
        super().__init__(fitness_function, bitstring_size, minmax_problem, random_walk, noimprove, searchspace=searchspace, neighbour=neighbour)
        self.restart = restart_search

    def hillclimb_candidate(self, maxfeval):
        self.current_candidate, extra_fevals, totfit, self.visited_cache = hillclimb.RandomGreedyHillclimb(self.current_candidate, self.ffunc, self.minmax, self.func_evals, maxfeval, self.visited_cache, self.nbour_method, restart=self.restart)
        self.func_evals += extra_fevals
        self.cumulative_fit += totfit

class OrderedGreedyILS(iterative_local_search_base):
    def __init__(self, fitness_function, bitstring_size, minmax_problem, random_walk, noimprove=100, searchspace=None, neighbour='Hamming', restart_search=True, order=None):
        r"""
            Ordered Greedy ILS which goes through all neighbours in order and
              moves to the first improvement.

        Args:
            iterations (int): Number of random restarts.
            fitness_function (bitarray() -> float): Function that
                scores fitness of bitstrings.
            bitstring_size (int): Length of the bitstring instances.
            min_max_problem (int): 1 if maximization problem, -1 for
                    minimization problem. Default is 1.
            random_walk (int): Length of random walk to generate
                new starting candidate.
            noimprove (int): (optional) Terminate if no improvement
                after this many trials. Default is 100.
            searchspace (dict): Mapping of settings to fitnesses
            neighbour (string): Method for generating neighbour solutions to visit.
            restartsearch (bool): Bool to decide whether we keep searching in the loop,
                    or break out of the loop en hillclimb from the start
            order (list): Order in which the variables should be
                traversed to find improvement neighbours.
        """
        super().__init__(fitness_function, bitstring_size, minmax_problem, random_walk, noimprove, searchspace=searchspace, neighbour=neighbour)
        self.restart = restart_search
        self.order = order

    def hillclimb_candidate(self, maxfeval):
        self.current_candidate, extra_fevals, totfit, self.visited_cache = hillclimb.OrderedGreedyHillclimb(self.current_candidate, self.ffunc, self.minmax, self.func_evals, maxfeval, self.visited_cache, self.nbour_method, self.order, restart=self.restart)
        self.func_evals += extra_fevals
        self.cumulative_fit += totfit

class StochasticILS(iterative_local_search_base):
    def __init__(self, fitness_function, bitstring_size, minmax_problem, random_walk, noimprove=100, searchspace=None, neighbour='Hamming'):
        r"""
            Stochastic improvement ILS which goes through all neighbours
             and moves to possible improvement with a probability that 
             depends on the size of the improvement.

        Args:
            iterations (int): Number of random restarts.
            fitness_function (bitarray() -> float): Function that
                scores fitness of bitstrings.
            bitstring_size (int): Length of the bitstring instances.
            min_max_problem (int): 1 if maximization problem, -1 for
                    minimization problem. Default is 1.
            random_walk (int): Length of random walk to generate
                new starting candidate.
            noimprove (int): (optional) Terminate if no improvement
                after this many trials. Default is 100.
            searchspace (dict): Mapping of settings to fitnesses
            neighbour (string): Method for generating neighbour solutions to visit.
        """
        super().__init__(fitness_function, bitstring_size, minmax_problem, random_walk, noimprove, searchspace=searchspace, neighbour=neighbour)

    def hillclimb_candidate(self, maxfeval):
        self.current_candidate, extra_fevals, totfit, self.visited_cache = hillclimb.StochasticHillclimb(self.current_candidate, self.ffunc, self.minmax, self.func_evals, maxfeval, self.visited_cache, self.nbour_method)
        self.func_evals += extra_fevals
        self.cumulative_fit += totfit
