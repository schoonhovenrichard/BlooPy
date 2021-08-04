import random
import sys
from timeit import default_timer as timer
import copy
import numpy as np
from bitarray import bitarray
import math
import scipy.optimize as scop
import scipy.stats

import bloopy.utils as utils
from bloopy.individual import continuous_individual
from bloopy.algorithms.continuous_base import continuous_base


def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return scipy.stats.truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


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

    def __call__(self, xk):
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

class local_minimizer(continuous_base):
    def __init__(self, fitness_function, minmax_problem, searchspace, method='COBYLA'):
        r"""
        Base Differential Evolutions algorithm.

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
        super().__init__(fitness_function,
                minmax_problem,
                searchspace)
        self.method = method

        supported_methods = ["Nelder-Mead", "Powell", "CG", "L-BFGS-B", "COBYLA", "SLSQP", "BFGS"]
        if self.method not in supported_methods:
            raise Exception("Unknown method passed to local minimizer!")
        self.get_scaling()

    def get_x0(self, bounds):
        x0 = []
        for xmin, xmax in bounds:
            x0.append(get_truncated_normal(mean=(xmax+xmin)/float(2.0), sd=(xmax-xmin)/float(100.0), low=xmin, upp=xmax).rvs())
        #values = self.sspace.values()
        #x0 = [0.5*self.eps*len(v) for v in values]
        return np.array(x0)

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
            raise Exception("Callback to break computation Basin Hopping")
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
        tindiv = continuous_individual(len(self.boundary_list)*[0.0], self.sspace, scaling=self.eps)
        self.maxf = max_funcevals
        self.solution = None
        self.solution_fit = None

        options = dict()
        bnds = self.get_scaling()

        kwargs = dict()
        if self.method in ["L-BFGS-B", "SLSQP"]:
            lb = np.array(bnds)[:,0]
            ub = np.array(bnds)[:,1]
            bounds = scipy.optimize.Bounds(lb, ub)
            kwargs['bounds'] = bounds
        if self.method in ["CG", "L-BFGS-B", "SLSQP", "BFGS"]:
            options['eps'] = self.eps
        elif self.method == "COBYLA":
            options['rhobeg'] = self.eps
        self.stopfit = stopping_fitness

        x0 = self.get_x0(bnds)

        try:
            with utils.suppress_stdout_stderr():
                solution = scop.minimize(self.cost_func, x0, method=self.method, options=options, **kwargs, callback=StopAlgorithm(max_time, self.ffunc, stopping_fitness, tindiv, self))
        finally:
            float_indiv = continuous_individual(self.solution, self.sspace, scaling=self.eps)
            float_indiv.fitness = -1 * self.minmax * self.solution_fit
            return (float_indiv.fitness, float_indiv, self.nfeval)
