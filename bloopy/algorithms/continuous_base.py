import random
import sys
from timeit import default_timer as timer
import copy
import numpy as np
from bitarray import bitarray

import bloopy.utils as utils
from bloopy.individual import continuous_individual


class continuous_base:
    def __init__(self, fitness_function, minmax_problem, searchspace):
        r"""
        Base class for all optimizers that work on real-valued vector
          solutions.

        Args:
            fitness_function (bitarray() -> float): Function that
                scores fitness of bitstrings.
            min_max_problem (int): 1 if maximization problem, -1 for
                    minimization problem. Default is 1.
            searchspace (dict(params, vals)): Dict that contains the
                tunable variables and their possible values.
        """
        self.ffunc = fitness_function
        self.minmax = minmax_problem
        self.sspace = searchspace
        self.boundary_list = utils.generate_boundary_list(self.sspace)
        self.visited_cache = dict()
        self.nfeval = 0

    def get_bounds(self):
        r""" Create a list bounds [(min,max)] for each entry in
            the multivariable vector.
        """
        bounds = []
        for values in self.sspace.values():
            sorted_values = np.sort(values)
            bounds.append((sorted_values[0], sorted_values[-1]))
        return bounds

    def get_scaling(self):
        values = self.sspace.values()
        eps = np.amin([1.0/float(len(v)) for v in values])
        self.eps = eps

        scaled_bounds = [(0.0, eps*len(v)) for v in values]
        orig_bounds = self.get_bounds()
        return scaled_bounds

    def cost_func(self, y):
        r"""
        Cost function to optimize.
        """
        float_indiv = continuous_individual(y, self.sspace, scaling=self.eps)
        bsstr = float_indiv.bitstring.to01()
        if bsstr in self.visited_cache:
            return self.visited_cache[bsstr]
        else:
            fit = self.ffunc(float_indiv.bitstring)
            self.nfeval += 1
            self.visited_cache[bsstr] = fit
            return fit
