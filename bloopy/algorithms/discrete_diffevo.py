import random
import sys
from timeit import default_timer as timer
import copy
import numpy as np

import bloopy.utils as utils
from bloopy.algorithms.genetic_algorithm import genetic_algorithm
import bloopy.selection_functions as sel
import bloopy.reproductive_functions as rep
import bloopy.fitness_functions as ff
from bloopy.individual import individual


class discrete_diffevo(genetic_algorithm):
    def __init__(self,
            fitness_function,
            population_size,
            bitstring_size,
            min_max_problem=1, #1 for maximization problems, -1 for minimization
            searchspace=None,
            input_pop=None,
            mutrange=(0.2,0.7),
            recomb=0.9):
        r"""
        Base Linkage Tree Genetic Algorithm. Most functionalities can
            be adapted by changing input component functions.

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
                rep.mask_crossover_pair, # The reproductor is fixed now
                None, # Selector is not used
                population_size,
                bitstring_size,
                min_max_problem,
                searchspace,
                input_pop,
                None)#Mutations are not used
        self.mutrange = mutrange
        self.recomb = recomb

    def one_generation(self):
        r"""
        Perform one generation of Linkage Tree GA.
        """
        children = self.create_offspring(self.current_pop)
        self.current_pop = children

    def create_offspring2(self, parents):
        r"""
        Create offspring solutions. Uses mask crossover and
            mutual information between bits.
        """
        children = []
        if len(parents) % 2 != 0:
            raise Exception("Not implemented for uneven number of parents!")
        best = self.current_best()
        random.shuffle(parents)
    
        mut = (self.mutrange[1]-self.mutrange[0])*random.random() + self.mutrange[0]
        for k in range(len(parents)):
            a = random.randint(0, len(parents)-1)
            r1 = parents[a]

            # Random mutation constant

            bprime = individual(best.size, bitstring=copy.deepcopy(best.bitstring), boundary_list=best.boundary_list)
            if bprime.boundary_list is not None:
                indicesb = [i for i, x in enumerate(list(best.bitstring)) if x]
                indices1 = [i for i, x in enumerate(list(r1.bitstring)) if x]
                for p in range(len(indicesb)):
                    if indicesb[p] == indices1[p]:
                        #If they are equal, we take bests gene
                        continue
                    else:
                        if random.random() < mut:
                            continue
                        else:
                            posb = indicesb[p]
                            pos1 = indices1[p]
                            bprime.bitstring[posb] = 0
                            bprime.bitstring[pos1] = 1
                #trial creation
                indicesbp = [i for i, x in enumerate(list(bprime.bitstring)) if x]
                indicesk = [i for i, x in enumerate(list(parents[k].bitstring)) if x]
                j = random.randint(0, len(indicesbp)-1)
                for p in range(len(indicesbp)):
                    if p == j:
                        continue
                    if random.random() > self.recomb:
                        posbp = indicesbp[p]
                        posk = indicesk[p]
                        bprime.bitstring[posbp] = 0
                        bprime.bitstring[posk] = 1
            else:
                for b in range(best.size):
                    if r1.bitstring[b] == r2.bitstring[b]:
                        continue
                    else:
                        if random.random() < mut:
                            continue
                        else:
                            if random.random() < 0.5:
                                bprime.bitstring[b] = r1.bitstring[b]
                            else:
                                bprime.bitstring[b] = r2.bitstring[b]
                #trial creation
                j = random.randint(0, bprime.size - 1)
                for b in range(bprime.size):
                    if b == j:
                        continue
                    if random.random() > self.recomb:
                        bprime.bitstring[b] = parents[k].bitstring[b]

            bsstr = bprime.bitstring.to01()
            if bsstr in self.visited_cache:
                bprime.fitness = self.visited_cache[bsstr]
            else:
                bprime.fitness = self.ffunc(bprime.bitstring)
                self.visited_cache[bsstr] = bprime.fitness
                self.func_evals += 1

            # If better, replace candidate
            if self.minmax * parents[k].fitness < self.minmax * bprime.fitness:
                children.append(bprime)
            else:
                children.append(parents[k])
            # If better, replace best by bprime
            if self.minmax * best.fitness < self.minmax * bprime.fitness:
                best = bprime
        return children

    def create_offspring(self, parents):
        r"""
        Create offspring solutions. Uses mask crossover and
            mutual information between bits.
        """
        children = []
        if len(parents) % 2 != 0:
            raise Exception("Not implemented for uneven number of parents!")
        best = self.current_best()
        random.shuffle(parents)
    
        mut = (self.mutrange[1]-self.mutrange[0])*random.random() + self.mutrange[0]
        for k in range(len(parents)):
            a = random.randint(0, len(parents)-1)
            b = random.randint(0, len(parents)-1)
            r1 = parents[a]
            r2 = parents[b]

            # Random mutation constant

            bprime = individual(best.size, bitstring=copy.deepcopy(best.bitstring), boundary_list=best.boundary_list)
            if bprime.boundary_list is not None:
                indicesb = [i for i, x in enumerate(list(best.bitstring)) if x]
                indices1 = [i for i, x in enumerate(list(r1.bitstring)) if x]
                indices2 = [i for i, x in enumerate(list(r2.bitstring)) if x]
                for p in range(len(indicesb)):
                    if indices1[p] == indices2[p]:
                        #If they are equal, we take bests gene
                        continue
                    else:
                        if random.random() < mut:
                            continue
                        else:
                            posb = indicesb[p]
                            bprime.bitstring[posb] = 0
                            if random.random() < 0.5:
                                pos1 = indices1[p]
                                bprime.bitstring[pos1] = 1
                            else:
                                pos2 = indices2[p]
                                bprime.bitstring[pos2] = 1
                #trial creation
                indicesbp = [i for i, x in enumerate(list(bprime.bitstring)) if x]
                indicesk = [i for i, x in enumerate(list(parents[k].bitstring)) if x]
                j = random.randint(0, len(indicesbp)-1)
                for p in range(len(indicesbp)):
                    if p == j:
                        continue
                    if random.random() > self.recomb:
                        posbp = indicesbp[p]
                        posk = indicesk[p]
                        bprime.bitstring[posbp] = 0
                        bprime.bitstring[posk] = 1
            else:
                for b in range(best.size):
                    if r1.bitstring[b] == r2.bitstring[b]:
                        continue
                    else:
                        if random.random() < mut:
                            continue
                        else:
                            if random.random() < 0.5:
                                bprime.bitstring[b] = r1.bitstring[b]
                            else:
                                bprime.bitstring[b] = r2.bitstring[b]
                #trial creation
                j = random.randint(0, bprime.size - 1)
                for b in range(bprime.size):
                    if b == j:
                        continue
                    if random.random() > self.recomb:
                        bprime.bitstring[b] = parents[k].bitstring[b]

            bsstr = bprime.bitstring.to01()
            if bsstr in self.visited_cache:
                bprime.fitness = self.visited_cache[bsstr]
            else:
                bprime.fitness = self.ffunc(bprime.bitstring)
                self.visited_cache[bsstr] = bprime.fitness
                self.func_evals += 1

            # If better, replace candidate
            if self.minmax * parents[k].fitness < self.minmax * bprime.fitness:
                children.append(bprime)
            else:
                children.append(parents[k])
            # If better, replace best by bprime
            if self.minmax * best.fitness < self.minmax * bprime.fitness:
                best = bprime
        return children
