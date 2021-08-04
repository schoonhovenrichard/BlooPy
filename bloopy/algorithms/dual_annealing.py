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

    def __call__(self, xk, f, context):
        elapsed = timer() - self.start
        self.temp_indiv.float_solution = xk
        self.temp_indiv.set_bitstring()
        nfevals = self.parent.nfeval

        bestfit = self.ffunc(self.temp_indiv.bitstring)
        if self.stop_fit is not None and bestfit*self.parent.minmax >= self.stop_fit*self.parent.minmax:
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


class dual_annealing(continuous_base):
    def __init__(self, fitness_function, minmax_problem, searchspace, method = 'BFGS'):
        r"""
        Base Dual Annealing algorithm.

        Args:
            fitness_function (bitarray() -> float): Function that
                scores fitness of bitstrings.
            bitstring_size (int): Length of the bitstring instances.
            min_max_problem (int): 1 if maximization problem, -1 for
                    minimization problem. Default is 1.
            searchspace (dict(params, vals)): Dict that contains the
                tunable variables and their possible values.
        """
        super().__init__(fitness_function,
                minmax_problem,
                searchspace)
        self.method = method
        supported_methods = ['COBYLA','L-BFGS-B','SLSQP','CG','Powell','Nelder-Mead', 'BFGS', 'trust-constr']
        if self.method not in supported_methods:
            raise Exception("Unknown method passed as local optimizer!")

    def cost_func(self, y):
        r"""
        Intermediate function to supply to scipy.optimize function.
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
            raise Exception("Callback to break computation DSA")
        return fit

    def solve(self,
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
        self.stopfit = stopping_fitness

        options = dict()
        options['method'] = self.method

        tindiv = continuous_individual(nr_vals*[0.0], self.sspace, scaling=self.eps)
        self.solution = None
        self.solution_fit = None

        try:
            solution = scop.dual_annealing(self.cost_func, bounds, maxiter=max_iter, callback=StopAlgorithm(max_time, self.ffunc, stopping_fitness, tindiv, self), local_search_options=options)
        finally:
            float_indiv = continuous_individual(self.solution, self.sspace, scaling=self.eps)
            float_indiv.fitness = -1 * self.minmax * self.solution_fit
            return (float_indiv.fitness, float_indiv, self.nfeval)
