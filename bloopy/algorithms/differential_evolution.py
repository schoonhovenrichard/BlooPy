import random
import sys
from timeit import default_timer as timer
import copy
import numpy as np
from bitarray import bitarray
import scipy.optimize as scop

import bloopy.utils as utils
from bloopy.individual import continuous_individual
from bloopy.algorithms.continuous_base import continuous_base

class StopAlgorithm(object):
    def __init__(self, max_time, fitness_func, stop_fit, indiv, parent):
        r"""
        Wrapper object to be supplied as callback= argument for
            scipy.optimize algorithm to regulate termination.
        """
        self.max_time = max_time
        self.start = timer()
        self.ffunc = fitness_func
        self.stop_fit = stop_fit
        self.temp_indiv = indiv
        self.parent = parent

    def __call__(self, xk=None, convergence=None):
        elapsed = timer() - self.start
        self.temp_indiv.float_solution = xk
        self.temp_indiv.set_bitstring()
        nfevals = self.parent.nfeval

        bestfit = self.ffunc(self.temp_indiv.bitstring)
        if self.stop_fit is not None and self.parent.minmax*bestfit >= self.parent.minmax * self.stop_fit:
            print("Terminating optimization: optimal solution reached", flush=True)
            return True
        elif nfevals >= self.parent.maxf:
            print("Terminating optimization: max funcevals reached", flush=True)
            return True
        elif elapsed > self.max_time:
            print("Terminating optimization: time limit reached", flush=True)
            return True
        else:
            print("Elapsed: %.3f sec" % elapsed, end="\r", flush=True)
            return False


class differential_evolution(continuous_base):
    def __init__(self, fitness_function, minmax_problem, searchspace, method=None, hillclimb=False, pop_size=15, mutation=(0.5,1), recombination=0.7):
        r"""
        Base Differential Evolutions algorithm.

        Args:
            fitness_function (bitarray() -> float): Function that
                scores fitness of bitstrings.
            min_max_problem (int): 1 if maximization problem, -1 for
                    minimization problem. Default is 1.
            searchspace (dict(params, vals)): Dict that contains the
                tunable variables and their possible values.
            method (string): (optional) scipy.optimize method to use.
            hillclimb (bool): (optional) Should solution be hillclimbed
                afterwards (calls polish=True for scipy.optimize).
                Default is False.
            pop_size (int): (optional) Populations size, default is 15.
        """
        super().__init__(fitness_function,
                minmax_problem,
                searchspace)
        self.method = method
        self.hillclimb = hillclimb
        self.pop_size = pop_size
        self.mutation = mutation
        self.recombination = recombination

        self.supported_methods = ["best1bin", "best1exp", "rand1exp", "randtobest1exp", "best2exp", "rand2exp", "randtobest1bin", "best2bin", "rand2bin", "rand1bin"]
        if self.method is None:
            self.method = self.supported_methods[0]

        self.solution = None
        self.solution_fit = None

    def cost_func(self, y):
        r"""
        Cost function to optimize.
        NOTE: The "-1 *" is because differential evolution only does maximization
        """
        float_indiv = continuous_individual(y, self.sspace, scaling=self.eps)
        bsstr = float_indiv.bitstring.to01()
        if bsstr in self.visited_cache:
            fit = self.visited_cache[bsstr]
        else:
            # These optimizers do only minimization problems, for maximization,
            #  we flip the fitness to negative for it to work.
            fit = -1 * self.minmax * self.ffunc(float_indiv.bitstring)
            self.nfeval += 1
            self.visited_cache[bsstr] = fit
        if self.solution_fit is None or fit < self.solution_fit:
            self.solution = y
            self.solution_fit = fit
        if self.nfeval >= self.maxf or self.solution_fit <= -1*self.stopfit*self.minmax:
            raise Exception("Callback to break computation Differential evolution")
        return fit

    def solve(self,
            min_variance, # Stopping population variance
            max_iter, # Max number of generations run
            max_time, # Max running time in seconds
            stopping_fitness, # If we know the best solution, may as well stop early
            max_funcevals=None
            ):
        r"""
        Solve problem using the algorithm until certain conditions are met.

        Args:
            min_variance (float): Stopping variance of population.
            max_time (int): Max running time in seconds.
            max_iter (int): Number of iterations to run.
            stopping_fitness (float): Stop evaluation if this fitness is
                reached. If we do not know, put +-np.inf.
            max_funcevals (int): (optional) Maximum number of fitness
                function evaluations before terminating.

        Returns tuple of:
            best_fit (float): Best fitness reached.
            self.best_candidate (individual): Best individual found.
            self.func_evals (int): Total number of Fevals performed.
        """
        bounds = self.get_scaling()
        nr_vals = len(list(self.sspace.values()))
        self.maxf = max_funcevals
        tindiv = continuous_individual(nr_vals*[0.0], self.sspace, scaling=self.eps)
        self.stopfit = stopping_fitness

        try:
            solution = scop.differential_evolution(self.cost_func, bounds, popsize=self.pop_size, maxiter=max_iter, tol=min_variance, callback=StopAlgorithm(max_time, self.ffunc, stopping_fitness, tindiv, self), polish=self.hillclimb, strategy=self.method, mutation=self.mutation, recombination=self.recombination)
        finally:
            float_indiv = continuous_individual(self.solution, self.sspace, scaling=self.eps)
            float_indiv.fitness = -1 * self.minmax * self.solution_fit
            return (float_indiv.fitness, float_indiv, self.nfeval)
