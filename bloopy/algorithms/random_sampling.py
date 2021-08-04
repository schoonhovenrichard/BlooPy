import random
import sys
from timeit import default_timer as timer
import copy
import numpy as np
from bitarray import bitarray

import bloopy.utils as utils
from bloopy.individual import individual


class random_sampling:
    def __init__(self, fitness_function, bitstring_size, minmax_problem, searchspace=None):
        r"""
        Random sampling algorithm. Randomly draws examples from the searchspace
            until the maximum number of evaluations has been reached. Method
            caches previous draws.

        Args:
            fitness_function (bitarray() -> float): Function that
                scores fitness of bitstrings.
            bitstring_size (int): Length of the bitstring instances.
            min_max_problem (int): 1 if maximization problem, -1 for
                    minimization problem. Default is 1.
            searchspace (dict(params, vals)): Dict that contains the
                tunable variables and their possible values.
            boundary_list (list(tuple(int))): (optional) None if 
                regular bitstrings. Otherwise, list of tuples 
                (start, end) of each segment of the bitstring in
                which we can have only one 1 that points to the
                element of the list that is active.
        """
        self.ffunc = fitness_function
        self.minmax = minmax_problem
        self.bs_size = bitstring_size
        self.sspace = searchspace
        if searchspace is None:
            self.boundary_list = None
        else:
            self.boundary_list = utils.generate_boundary_list(searchspace)
    
    def solve(self,
            max_time, # Max running time in seconds
            stopping_fitness, # If we know the best solution, may as well stop early
            max_funcevals=None,
            verbose=False,
            ):
        r"""
        Solve problem using the algorithm until certain conditions are met.

        Args:
            min_variance (float): Stopping variance of population.
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
        begintime = timer()
        previous_draws = []
        candidate = individual(self.bs_size, boundary_list=self.boundary_list)
        best_candidate = None
        max_redraws = 10
        for evl in range(max_funcevals):
            candidate.generate_random(self.boundary_list)
            cache_redraws = 0
            while candidate.bitstring in previous_draws:
                candidate.generate_random(self.boundary_list)
                cache_redraws += 1
                if cache_redraws >= max_redraws:
                    break
            previous_draws.append(candidate.bitstring)
            candidate.fitness = self.ffunc(candidate.bitstring)

            # Record best draw
            if best_candidate is None or best_candidate.fitness*self.minmax < candidate.fitness*self.minmax:
                best_candidate = copy.deepcopy(candidate)

            # Stopping conditions
            end = timer()
            elapsed_time = end - begintime
            if elapsed_time > max_time:
                break
            if verbose:
                print('Running iteration {0} |Elapsed (s): {1:.2f} |Best fit: {2:.1f} |\r'.format(evl, elapsed_time, best_candidate.fitness), end="")

            if stopping_fitness is not None:
                if self.minmax * best_candidate.fitness >= self.minmax * stopping_fitness:
                    break
        return (best_candidate.fitness, best_candidate, max_funcevals)
