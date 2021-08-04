import random
import sys
from timeit import default_timer as timer
import copy
import numpy as np
from bitarray import bitarray
import scipy.optimize as scop

import pyswarms
from pyswarms.backend.topology import Ring

from bloopy.individual import continuous_individual
import bloopy.utils as utils
from bloopy.algorithms.continuous_base import continuous_base


class pyswarms_pso(continuous_base):
    def __init__(self, fitness_function, minmax_problem, searchspace, n_particles=100, w=0.5, c1=2, c2=1, k=10, p=1, topology=Ring(), scaling=None):
        r"""
        Base Particle Swarm Algorithm using PySwarms.

        Args:
            fitness_function (bitarray() -> float): Function that
                scores fitness of bitstrings.
            min_max_problem (int): 1 if maximization problem, -1 for
                    minimization problem. Default is 1.
            searchspace (dict(params, vals)): Dict that contains the
                tunable variables and their possible values.
            n_particles (int): (optional) Number of swarm particles, default is 100.
            w (float): Intertia constant.
            c1 (float): Cognitive constant.
            c2 (float): Socal constant.
            k (int): Number of neighbours to be considered.
            p (int): Minskowski p-norm for topology.
        """
        super().__init__(fitness_function,
                minmax_problem,
                searchspace)
        self.nparts = n_particles
        self.params = dict()
        self.params['w'] = w
        self.params['c1'] = c1
        self.params['c2'] = c2
        self.params['k'] = k
        self.params['p'] = p
        self.topo = topology
        self.scaling = scaling
        self.eps = None

        if self.scaling is not None:
            bounds = self.get_scaling()
            minbounds = np.array(bounds)[:,0]
            maxbounds = np.array(bounds)[:,1]
            boundsnp =  tuple([minbounds, maxbounds])
        else:
            bounds = self.get_bounds()
            minbounds = []
            maxbounds = []
            for i in range(len(bounds)):
                minbounds.append(bounds[i][0])
                maxbounds.append(bounds[i][1])
            boundsnp = tuple([np.array(minbounds), np.array(maxbounds)])
        self.pso = pyswarms.single.general_optimizer.GeneralOptimizerPSO(self.nparts, len(bounds), self.params, topology=self.topo, bounds=boundsnp)

    def cost_func(self, y, kwargs):
        fitnesses  = []
        fitness_func = kwargs['fitness_func']
        searchspace = kwargs['searchspace']
        scaling = kwargs['eps']
        for k in range(y.shape[0]):
            float_indiv = continuous_individual(y[k], searchspace, scaling=scaling)
            bsstr = float_indiv.bitstring.to01()
            if bsstr in self.visited_cache:
                fit = self.visited_cache[bsstr]
            else:
                # These optimizers do only minimization problems, for maximization,
                #  we flip the fitness to negative for it to work.
                fit = -1 * self.minmax * self.ffunc(float_indiv.bitstring)
                self.nfeval += 1
                self.visited_cache[bsstr] = fit
            fitnesses.append(fit)
        if self.nfeval >= self.maxf:
            raise Exception("Callback to break computation pyswarms")
        return np.array(fitnesses)

    def get_scaling(self):
        values = self.sspace.values()
        eps = self.scaling * np.amin([1.0/float(len(v)) for v in values])
        self.eps = eps

        scaled_bounds = [(0.0, eps*len(v)) for v in values]
        orig_bounds = self.get_bounds()
        return scaled_bounds

    def solve(self,
            max_iter, # Max number of generations run
            max_funcevals=None,
            n_procs=None):
        r"""
        Solve problem using the algorithm until certain conditions are met.

        Args:
            max_iter (int): Number of iterations to run.
            n_procs (int): (optional) Number of processors to use in parallel.

        Returns tuple of:
            best_fit (float): Best fitness reached.
            self.best_candidate (individual): Best individual found.
            self.func_evals (int): Total number of Fevals performed.
        """
        self.maxf = max_funcevals
        arguments = dict()
        arguments['fitness_func'] = self.ffunc
        arguments['searchspace'] = self.sspace
        arguments['eps'] = self.eps

        try:
            solution = self.pso.optimize(self.cost_func, max_iter, n_processes=n_procs, kwargs=arguments, verbose=False)
        except:
            # We still have to update the last run, otherwise its unfair
            final_best_cost = self.pso.swarm.best_cost
            final_best_pos = self.pso.swarm.pbest_pos[self.pso.swarm.pbest_cost.argmin()].copy()
            parts = self.pso.swarm.position
            best_part = None
            best_fit = None
            for x in parts:
                float_indiv = continuous_individual(x, self.sspace, scaling=self.eps)
                bsstr = float_indiv.bitstring.to01()
                if bsstr in self.visited_cache:
                    fit = self.visited_cache[bsstr]
                    # Because fitness is negative for maximization problems,
                    #   this auto works
                    if best_fit is None or fit < best_fit:
                        best_fit = fit
                        best_part = x
                else:
                    raise Exception("Whats going on here?")
            # Because fitness is negative for maximization problems, this auto works
            if best_fit < final_best_cost:
                solution = (best_fit, best_part)
            else:
                solution = (final_best_cost, final_best_pos)

        float_indiv = continuous_individual(solution[1].tolist(), self.sspace, scaling=self.eps)
        float_indiv.fitness = -1 * self.minmax * solution[0]
        return (float_indiv.fitness, float_indiv, self.nfeval)
