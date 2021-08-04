import random
import sys
from timeit import default_timer as timer
import copy
import numpy as np
import multiprocessing as mp

import bloopy.utils as utils
from bloopy.algorithms.ltga import ltga
import bloopy.reproductive_functions as rep


class ltga_greedy(ltga):
    def __init__(self,
            fitness_function,
            population_size,
            bitstring_size,
            min_max_problem=1, #1 for maximization problems, -1 for minimization
            searchspace=None,
            input_pop=None,
            maxdepth=None):
        r"""
        Base Linkage Tree Genetic Algorithm with greedy selection.

        Args:
            fitness_function (bitarray() -> float): Function that
                scores fitness of bitstrings.
            population_size (int): Size of population of solutions.
            bitstring_size (int): Length of the bitstring instances.
            min_max_problem (int): 1 if maximization problem, -1 for
                    minimization problem. Default is 1.
            boundary_list (list(tuple(int))): (optional) None if 
                regular bitstrings. Otherwise, list of tuples 
                (start, end) of each segment of the bitstring in
                which we can have only one 1 that points to the
                element of the list that is active.
            input_pop (list(individual)): (optional) Possible input
                population from which to start algorithm. If None,
                the GA will generate its own.
            maxdepth (int): Maximum tree depth for search masks
                for mutual information.
        """
        super().__init__(fitness_function,
                population_size,
                bitstring_size,
                min_max_problem,
                searchspace,
                input_pop,
                maxdepth)
        self.reproductor = rep.greedy_mask_crossover_pair

    def create_offspring(self, parents):
        r"""
        Create offspring for LTGA greedily by selecting random donor 
        for mask crossover. Read paper for further details
        """
        self.build_linkage_tree()
        children = []
        random.shuffle(parents)
        for i in range(self.pop_size):
            current_sol = parents[i]
            for j in range(len(self.linkage_tree)):
                crossover_mask = self.linkage_tree[j]
                rand = random.randint(0, self.pop_size-1)
                while rand == i:
                    rand = random.randint(0, self.pop_size-1)
                donor = parents[rand]
                child = self.reproductor(current_sol, donor, crossover_mask)
                if child.bitstring == current_sol.bitstring:
                    continue

                if self.LOGTOLIN:
                    self.set_fitness_log(child)
                else:
                    self.set_fitness(child)

                if child.fitness*self.minmax > current_sol.fitness*self.minmax:
                    current_sol = child
            children.append(current_sol)
        return children
