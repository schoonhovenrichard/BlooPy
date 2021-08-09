import random
import sys
from timeit import default_timer as timer
import copy
import numpy as np
from bitarray import bitarray

import bloopy.utils as utils
from bloopy.individual import individual
import bloopy.algorithms.hillclimbers as hillclimb

class multi_start_local_search_base:
    def __init__(self, fitness_function, bitstring_size, minmax_problem, searchspace=None, neighbour='Hamming'):
        r"""
        Base MultiStart Local Search algorithm. Different child classes
            can be constructed which override hillcimb_candidate and 
            point_mutate.

        Args:
            iterations (int): Number of random restarts.
            fitness_function (bitarray() -> float): Function that
                scores fitness of bitstrings.
            bitstring_size (int): Length of the bitstring instances.
            min_max_problem (int): 1 if maximization problem, -1 for
                    minimization problem. Default is 1.
            searchspace (dict): Mapping of settings to fitnesses
            neighbour (string): Method for generating neighbour solutions to visit.
        """
        self.ffunc = fitness_function
        self.minmax = minmax_problem
        self.bs_size = bitstring_size
        if searchspace is None:
            self.boundary_list = None
        else:
            self.boundary_list = utils.generate_boundary_list(searchspace)
        self.best_candidate = None
        self.current_candidate = None
        self.func_evals = 0
        self.visited_cache = dict()
        if neighbour not in ['Hamming', 'adjacent']:
            raise Exception("Unknown implementation for neighbouring solutions")
        else:
            self.nbour_method = neighbour

    def generate_candidate(self, maxfeval):
        r"""
        Generate a new starting solution.
        """
        self.current_candidate = individual(self.bs_size, boundary_list=self.boundary_list)
        bsstr = self.current_candidate.bitstring.to01()
        if bsstr in self.visited_cache:
            self.current_candidate.fitness = self.visited_cache[bsstr]
        else:
            self.current_candidate.fitness = self.ffunc(self.current_candidate.bitstring)
            self.func_evals += 1
            self.visited_cache[bsstr] = self.current_candidate.fitness
        self.hillclimb_candidate(maxfeval)
        if self.best_candidate is None or self.best_candidate.fitness*self.minmax < self.current_candidate.fitness*self.minmax:
            self.best_candidate = self.current_candidate

    def point_mutate(self, indiv):
        r"""
        Perform a random point mutation on each individual.
        If self.boundary_list is not None, chooses a random segment,
            and selects a random element from it.
        """
        if self.boundary_list is None:
            pos = random.randint(0, len(indiv.bitstring)-1)
            indiv.bitstring[pos] = not indiv.bitstring[pos]
            indiv.fitness = None
        else:
            indices = [i for i, x in enumerate(list(indiv.bitstring)) if x]
            substr = random.randint(0, len(self.boundary_list)-1)
            indiv.bitstring[indices[substr]] = 0
            pos = random.randint(self.boundary_list[substr][0], self.boundary_list[substr][1])
            indiv.bitstring[pos] = 1
            indiv.fitness = None

    def solve(self, iters, max_time, stopping_fitness, max_funcevals=None, verbose=True):
        r"""
        Solve problem using the algorithm until certain conditions are met.

        Args:
            max_time (int): Max running time in seconds.
            stopping_fitness (float): Stop evaluation if this fitness is
                reached. If we do not know, put +-np.inf.
            max_funcevals (int): (optional) Maximum number of fitness
                function evaluations before terminating.
            verbose (bool): (optional) Run with the GA continually
                printing status updates. Default is True.

        Returns tuple of:
            best_fit (float): Best fitness reached.
            self.best_candidate (individual): Best individual found.
            self.func_evals (int): Total number of Fevals performed.
        """
        nonterminate = True
        begintime = timer()
        it = 0
        while it < iters and nonterminate:
            end = timer()
            self.generate_candidate(max_funcevals)
            best_fit = self.best_candidate.fitness
            it += 1

            elapsed_time = end - begintime
            if verbose:
                print('Running iteration {0} |Elapsed (s): {1:.2f} |Best fit: {2:.1f} |\r'.format(it, elapsed_time, best_fit), end="")

            if stopping_fitness is not None:
                if self.minmax * best_fit >= self.minmax * stopping_fitness:
                    nonterminate = False
                    if verbose:
                        print("Stopping fitness reached, terminating...")

            end = timer()
            elapsed_time = end - begintime
            if elapsed_time > max_time:
                nonterminate = False
                if verbose:
                    print("Max running time reached, terminating...")

            if max_funcevals is not None and self.func_evals >= max_funcevals:
                nonterminate = False
                if verbose:
                    print("{0} fitness evaluations reached in {1} iterations, terminating...".format(self.func_evals, it))

        if verbose:
            print("Terminated after {0} iterations with best fitness: {1:.3f} | # of fitness evals: {2}".format(it, best_fit, self.func_evals))
        return (best_fit, self.best_candidate, self.func_evals)

class OrderedGreedyMLS(multi_start_local_search_base):
    def __init__(self, fitness_function, bitstring_size, minmax_problem, searchspace=None, neighbour='Hamming', restart_search=True, order=None):
        r"""
            First improvement MLS which goes through neighbours in order.

        Args:
            iterations (int): Number of random restarts.
            fitness_function (bitarray() -> float): Function that
                scores fitness of bitstrings.
            bitstring_size (int): Length of the bitstring instances.
            min_max_problem (int): 1 if maximization problem, -1 for
                    minimization problem. Default is 1.
            searchspace (dict): Mapping of settings to fitnesses
            neighbour (string): Method for generating neighbour solutions to visit.
            restartsearch (bool): Bool to decide whether we keep searching in the loop,
                    or break out of the loop en hillclimb from the start
            order (list): Order in which the variables should be
                traversed to find improvement neighbours.
        """
        super().__init__(fitness_function, bitstring_size, minmax_problem, searchspace=searchspace, neighbour=neighbour)
        self.restart = restart_search
        self.order = order

    def hillclimb_candidate(self, maxfeval):
        self.current_candidate, extra_fevals, self.visited_cache = hillclimb.OrderedGreedyHillclimb(self.current_candidate, self.ffunc, self.minmax, self.func_evals, maxfeval, self.visited_cache, self.nbour_method, self.order, restart=self.restart)
        self.func_evals += extra_fevals

class RandomGreedyMLS(multi_start_local_search_base):
    def __init__(self, fitness_function, bitstring_size, minmax_problem, searchspace=None, neighbour='Hamming', restart_search=True):
        r"""
            First improvement MLS which goes through neighbours at random.

        Args:
            iterations (int): Number of random restarts.
            fitness_function (bitarray() -> float): Function that
                scores fitness of bitstrings.
            bitstring_size (int): Length of the bitstring instances.
            min_max_problem (int): 1 if maximization problem, -1 for
                    minimization problem. Default is 1.
            searchspace (dict): Mapping of settings to fitnesses
            neighbour (string): Method for generating neighbour solutions to visit.
            restartsearch (bool): Bool to decide whether we keep searching in the loop,
                    or break out of the loop en hillclimb from the start
        """
        super().__init__(fitness_function, bitstring_size, minmax_problem, searchspace=searchspace, neighbour=neighbour)
        self.restart = restart_search
    
    def hillclimb_candidate(self, maxfeval):
        self.current_candidate, extra_fevals, self.visited_cache = hillclimb.RandomGreedyHillclimb(self.current_candidate, self.ffunc, self.minmax, self.func_evals, maxfeval, self.visited_cache, self.nbour_method, restart=self.restart)
        self.func_evals += extra_fevals

class BestMLS(multi_start_local_search_base):
    def __init__(self, fitness_function, bitstring_size, minmax_problem, searchspace=None, neighbour='Hamming'):
        r"""
            Best improvement MLS which goes through all neighbours and moves to the best.

        Args:
            iterations (int): Number of random restarts.
            fitness_function (bitarray() -> float): Function that
                scores fitness of bitstrings.
            bitstring_size (int): Length of the bitstring instances.
            min_max_problem (int): 1 if maximization problem, -1 for
                    minimization problem. Default is 1.
            searchspace (dict): Mapping of settings to fitnesses
            neighbour (string): Method for generating neighbour solutions to visit.
        """
        super().__init__(fitness_function, bitstring_size, minmax_problem, searchspace=searchspace, neighbour=neighbour)

    def hillclimb_candidate(self, maxfeval):
        self.current_candidate, extra_fevals, self.visited_cache = hillclimb.BestHillclimb(self.current_candidate, self.ffunc, self.minmax, self.func_evals, maxfeval, self.visited_cache, self.nbour_method)
        self.func_evals += extra_fevals

class StochasticMLS(multi_start_local_search_base):
    def __init__(self, fitness_function, bitstring_size, minmax_problem, searchspace=None, neighbour='Hamming'):
        r"""
            Stochastic improvement MLS which goes through all neighbours
             and moves to possible improvement with a probability that 
             depends on the size of the improvement.

        Args:
            iterations (int): Number of random restarts.
            fitness_function (bitarray() -> float): Function that
                scores fitness of bitstrings.
            bitstring_size (int): Length of the bitstring instances.
            min_max_problem (int): 1 if maximization problem, -1 for
                    minimization problem. Default is 1.
            searchspace (dict): Mapping of settings to fitnesses
            neighbour (string): Method for generating neighbour solutions to visit.
        """
        super().__init__(fitness_function, bitstring_size, minmax_problem, searchspace=searchspace, neighbour=neighbour)

    def hillclimb_candidate(self, maxfeval):
        self.current_candidate, extra_fevals, self.visited_cache = hillclimb.StochasticHillclimb(self.current_candidate, self.ffunc, self.minmax, self.func_evals, maxfeval, self.visited_cache, self.nbour_method)
        self.func_evals += extra_fevals
