import random
from timeit import default_timer as timer
from bitarray import bitarray
import numpy as np
import copy

from bloopy.algorithms.genetic_algorithm import genetic_algorithm
from bloopy.individual import individual

class genetic_local_search(genetic_algorithm):
    def __init__(self,
            fitness_function,
            reproductor,
            selector,
            population_size,
            bitstring_size,
            hillclimber,
            min_max_problem=1, #1 for maximization problems, -1 for minimization
            searchspace=None,
            input_pop=None,
            neighbour="adjacent"):
        r"""
        Base genetic local search. Most functionalities can be adapted
            by changing input component functions.

        Args:
            fitness_function (bitarray() -> float): Function that
                scores fitness of bitstrings.
            reproductor: Function that creates offspring solutions.
            selector: Function that selects fittest individuals.
            population_size (int): Size of population of solutions.
            bitstring_size (int): Length of the bitstring instances.
            hillclimber (): Takes individual and hill climbs to a
                local optimum. Some implementations in hillclimbers.py
            min_max_problem (int): 1 if maximization problem, -1 for
                    minimization problem. Default is 1.
            searchspace (dict): Mapping of settings to fitnesses
            input_pop (list(individual)): (optional) Possible input
                population from which to start algorithm. If None,
                the GA will generate its own.
        """
        super().__init__(fitness_function,
                reproductor,
                selector,
                population_size,
                bitstring_size,
                min_max_problem=min_max_problem,
                searchspace=searchspace,
                input_pop=input_pop,
                mutation=None)#Mutations are not used
        self.hillclimber = hillclimber
        self.nbour_method = neighbour

    def one_generation(self):
        r"""
        Perform one generation of GLS.
        """
        parents = self.current_pop

        # Reproductive step
        children = self.create_offspring(parents)

        # Mutation step
        for i in range(len(children)):
            bsstr = children[i].bitstring.to01()
            if bsstr in self.visited_cache:
                children[i].fitness = self.visited_cache[bsstr]
            else:
                children[i].fitness = self.ffunc(children[i].bitstring)
                self.visited_cache[bsstr] = children[i].fitness
                self.cumulative_fit += children[i].fitness
                self.func_evals += 1

            children[i], feval, tot_fit, self.visited_cache = self.hillclimber(children[i], self.ffunc, self.minmax, self.func_evals, self.maxfeval, self.visited_cache, self.nbour_method)
            self.func_evals += feval
            self.cumulative_fit += tot_fit

        # Selection step
        self.current_pop = self.selector(parents, children, self.minmax)
