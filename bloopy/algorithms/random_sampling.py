import random
import sys
from timeit import default_timer as timer
import copy
import numpy as np
from bitarray import bitarray

import bloopy.utils as utils
from bloopy.individual import individual


class random_sampling:
    def __init__(self, fitness_function, bitstring_size, minmax_problem, searchspace=None, caching=True):
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
            caching (bool): If true, caches fitness for every point in search space
                    visited (repeated visits do not count towards function evaluation.
                    Should not be used for stochastic optimization.
        """
        self.ffunc = fitness_function
        self.minmax = minmax_problem
        self.bs_size = bitstring_size
        self.sspace = searchspace
        self.caching = caching
        self.func_evals = 0
        if searchspace is None:
            self.boundary_list = None
        else:
            self.boundary_list = utils.generate_boundary_list(searchspace)
        if self.caching:
            self.visited_cache = dict()
        else:
            self.visited_cache = None

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
        candidate = individual(self.bs_size, boundary_list=self.boundary_list)
        best_candidate = None
        for evl in range(max_funcevals):
            candidate.generate_random(self.boundary_list)

            bsstr = candidate.bitstring.to01()
            if self.caching:
                max_redraws = 10
                cache_redraws = 0
                while cache_redraws <= max_redraws:
                    if bsstr not in self.visited_cache:
                        break
                    cache_redraws += 1
                    candidate.generate_random(self.boundary_list)
                    bsstr = candidate.bitstring.to01()

            candidate.fitness = self.ffunc(candidate.bitstring)
            self.func_evals += 1
            if self.caching:
                bsstr = candidate.bitstring.to01()
                self.visited_cache[bsstr] = candidate.fitness

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
        return (best_candidate.fitness, best_candidate, self.func_evals)
