import random
import sys
from timeit import default_timer as timer
import copy
import numpy as np
from bitarray import bitarray
from collections import deque

from bloopy.individual import individual
from bloopy.algorithms.local_search import multi_start_local_search_base


class BestTabu(multi_start_local_search_base):
    def __init__(self, fitness_function, bitstring_size, min_max_problem, tabu_size, searchspace=None, neighbour="Hamming"):
        r"""
        Base Tabu Search algorithm.

        Args:
            iterations (int): Number of random restarts.
            fitness_function (bitarray() -> float): Function that
                scores fitness of bitstrings.
            bitstring_size (int): Length of the bitstring instances.
            min_max_problem (int): 1 if maximization problem, -1 for
                    minimization problem. Default is 1.
            tabu_size (int): Size of queue that maintains tabu solutions.
            searchspace (dict): Mapping of settings to fitnesses
            neighbour (string): Method for generating neighbour solutions to visit.
        """
        super().__init__(fitness_function, bitstring_size, min_max_problem, searchspace=searchspace, neighbour=neighbour)
        self.tabu_size = tabu_size
        self.tabu_list = deque(maxlen=tabu_size)

    def generate_candidate(self, maxfeval):
        r"""
        Generate a new candidate by flipping bits to find best
         neighbour solution. Solutions in Tabu queue are skipped.
        """
        # If we have no candidate yet
        if self.current_candidate is None:
            self.current_candidate = individual(self.bs_size, boundary_list=self.boundary_list)
            bsstr = self.current_candidate.bitstring.to01()
            if bsstr in self.visited_cache:
                self.current_candidate.fitness = self.visited_cache[bsstr]
            else:
                self.current_candidate.fitness = self.ffunc(self.current_candidate.bitstring)
                self.visited_cache[bsstr] = self.current_candidate.fitness
                self.func_evals += 1
                self.cumulative_fit += self.current_candidate.fitness
            if self.best_candidate is None or self.best_candidate.fitness*self.minmax < self.current_candidate.fitness*self.minmax:
                self.best_candidate = copy.deepcopy(self.current_candidate)

        best_neighbor = None
        neighbor = individual(self.current_candidate.size, bitstring=copy.deepcopy(self.current_candidate.bitstring), boundary_list=self.boundary_list)
        neighbor.fitness = self.current_candidate.fitness
        if self.boundary_list is None:
            # Do nothing with neighbour method for bitstrings
            for k in range(self.bs_size):
                if maxfeval is not None and self.func_evals >= maxfeval:
                    break
                neighbor.bitstring[k] = not neighbor.bitstring[k]
                if neighbor.bitstring in self.tabu_list:
                    # Skip this neighbour
                    neighbor.bitstring[k] = not neighbor.bitstring[k]
                    continue

                #Non tabu neighbour, get fitness
                bsstr = neighbor.bitstring.to01()
                if bsstr in self.visited_cache:
                    neighbor.fitness = self.visited_cache[bsstr]
                else:
                    neighbor.fitness = self.ffunc(neighbor.bitstring)
                    self.visited_cache[bsstr] = neighbor.fitness
                    self.func_evals += 1
                    self.cumulative_fit += neighbor.fitness

                if best_neighbor is None:
                    best_neighbor = copy.deepcopy(neighbor)
                elif self.minmax*neighbor.fitness > self.minmax*best_neighbor.fitness:
                    best_neighbor = copy.deepcopy(neighbor)

                neighbor.bitstring[k] = not neighbor.bitstring[k] # flip back
                neighbor.fitness = self.current_candidate.fitness
        else:
            indices = [i for i, x in enumerate(list(neighbor.bitstring)) if x]
            for k in range(len(self.boundary_list)):
                if self.nbour_method == "Hamming":
                    bs_idxs = list(range(self.boundary_list[k][0], self.boundary_list[k][1]+1))
                elif self.nbour_method == "adjacent":
                    if indices[k] == self.boundary_list[k][0]:
                        bs_idxs = [indices[k]+1]
                    elif indices[k] == self.boundary_list[k][1]:
                        bs_idxs = [indices[k]-1]
                    else:
                        bs_idxs = [indices[k]-1, indices[k]+1]
                for i in bs_idxs:
                    if i == indices[k]:
                        continue
                    if maxfeval is not None and self.func_evals >= maxfeval:
                        break
                    neighbor.bitstring[indices[k]] = 0 # Set old one to 0
                    neighbor.bitstring[i] = 1 # set new one to 1
                    if neighbor.bitstring in self.tabu_list:
                        # Skip this neighbour
                        neighbor.bitstring[i] = 0 # set new one to 0
                        neighbor.bitstring[indices[k]] = 1 # Set old one back to 1
                        continue

                    #Non tabu neighbour, get fitness
                    bsstr = neighbor.bitstring.to01()
                    if bsstr in self.visited_cache:
                        neighbor.fitness = self.visited_cache[bsstr]
                    else:
                        neighbor.fitness = self.ffunc(neighbor.bitstring)
                        self.visited_cache[bsstr] = neighbor.fitness
                        self.func_evals += 1
                        self.cumulative_fit += neighbor.fitness

                    if best_neighbor is None:
                        best_neighbor = copy.deepcopy(neighbor)
                    elif self.minmax*neighbor.fitness > self.minmax*best_neighbor.fitness:
                        best_neighbor = copy.deepcopy(neighbor)

                    neighbor.bitstring[i] = 0 # set new one to 0
                    neighbor.bitstring[indices[k]] = 1 # Set old one back to 1
                    neighbor.fitness = self.current_candidate.fitness

        if best_neighbor is None:
            # If the entire neighbourhood is tabu, return random point
            best_neighbor = individual(self.bs_size, boundary_list=self.boundary_list)
            bsstr = best_neighbor.bitstring.to01()
            if bsstr in self.visited_cache:
                best_neighbor.fitness = self.visited_cache[bsstr]
            else:
                best_neighbor.fitness = self.ffunc(best_neighbor.bitstring)
                self.visited_cache[bsstr] = best_neighbor.fitness
                self.func_evals += 1
                self.cumulative_fit += best_neighbor.fitness

        # We always set the best neighbour as current candidate, even if its worse
        self.tabu_list.append(best_neighbor.bitstring)
        self.current_candidate = best_neighbor
        if self.best_candidate is None or self.best_candidate.fitness*self.minmax < self.current_candidate.fitness*self.minmax:
            self.best_candidate = copy.deepcopy(self.current_candidate)

class RandomGreedyTabu(multi_start_local_search_base):
    def __init__(self, fitness_function, bitstring_size, min_max_problem, tabu_size, searchspace=None, neighbour="Hamming"):
        r"""
        Base Tabu Search algorithm.

        Args:
            iterations (int): Number of random restarts.
            fitness_function (bitarray() -> float): Function that
                scores fitness of bitstrings.
            bitstring_size (int): Length of the bitstring instances.
            min_max_problem (int): 1 if maximization problem, -1 for
                    minimization problem. Default is 1.
            tabu_size (int): Size of queue that maintains tabu solutions.
            searchspace (dict): Mapping of settings to fitnesses
            neighbour (string): Method for generating neighbour solutions to visit.
        """
        super().__init__(fitness_function, bitstring_size, min_max_problem, searchspace=searchspace, neighbour=neighbour)
        self.tabu_size = tabu_size
        self.tabu_list = deque(maxlen=tabu_size)

    def generate_candidate(self, maxfeval):
        r"""
        Generate a new candidate by flipping bits to find best
         neighbour solution. Solutions in Tabu queue are skipped.
        """
        # If we have no candidate yet
        if self.current_candidate is None:
            self.current_candidate = individual(self.bs_size, boundary_list=self.boundary_list)
            bsstr = self.current_candidate.bitstring.to01()
            if bsstr in self.visited_cache:
                self.current_candidate.fitness = self.visited_cache[bsstr]
            else:
                self.current_candidate.fitness = self.ffunc(self.current_candidate.bitstring)
                self.visited_cache[bsstr] = self.current_candidate.fitness
                self.func_evals += 1
                self.cumulative_fit += self.current_candidate.fitness
            if self.best_candidate is None or self.best_candidate.fitness*self.minmax < self.current_candidate.fitness*self.minmax:
                self.best_candidate = copy.deepcopy(self.current_candidate)

        best_neighbor = None
        neighbor = individual(self.current_candidate.size, bitstring=copy.deepcopy(self.current_candidate.bitstring), boundary_list=self.boundary_list)
        neighbor.fitness = self.current_candidate.fitness
        if self.boundary_list is None:
            # Do nothing with neighbour method for bitstrings
            shuffle = np.random.permutation(self.bs_size)
            for k in range(self.bs_size):
                if maxfeval is not None and self.func_evals >= maxfeval:
                    break
                idx = shuffle[k]
                neighbor.bitstring[idx] = not neighbor.bitstring[idx]
                if neighbor.bitstring in self.tabu_list:
                    # Skip this neighbour
                    neighbor.bitstring[idx] = not neighbor.bitstring[idx]
                    continue

                #Non tabu neighbour, get fitness
                bsstr = neighbor.bitstring.to01()
                if bsstr in self.visited_cache:
                    neighbor.fitness = self.visited_cache[bsstr]
                else:
                    neighbor.fitness = self.ffunc(neighbor.bitstring)
                    self.visited_cache[bsstr] = neighbor.fitness
                    self.func_evals += 1
                    self.cumulative_fit += neighbor.fitness

                # If we improved, immediately return
                if self.minmax*neighbor.fitness > self.minmax*self.current_candidate.fitness:
                    self.current_candidate = copy.deepcopy(neighbor)
                    self.tabu_list.append(neighbor.bitstring)
                    if self.best_candidate is None or self.best_candidate.fitness*self.minmax < self.current_candidate.fitness*self.minmax:
                        self.best_candidate = copy.deepcopy(self.current_candidate)
                    return

                # Update best neighbour
                if best_neighbor is None:
                    best_neighbor = copy.deepcopy(neighbor)
                elif self.minmax*neighbor.fitness > self.minmax*best_neighbor.fitness:
                    best_neighbor = copy.deepcopy(neighbor)

                neighbor.bitstring[idx] = not neighbor.bitstring[idx] # flip back
                neighbor.fitness = self.current_candidate.fitness
        else:
            indices = [i for i, x in enumerate(list(neighbor.bitstring)) if x]
            shuffle_pars = np.random.permutation(len(self.boundary_list)).tolist()
            for k in shuffle_pars:
                if self.nbour_method == "Hamming":
                    bs_idxs = np.arange(self.boundary_list[k][0], self.boundary_list[k][1]+1)
                    bs_idxs = np.random.permutation(bs_idxs).tolist()
                elif self.nbour_method == "adjacent":
                    if indices[k] == self.boundary_list[k][0]:
                        bs_idxs = [indices[k]+1]
                    elif indices[k] == self.boundary_list[k][1]:
                        bs_idxs = [indices[k]-1]
                    else:
                        bs_idxs = [indices[k]-1, indices[k]+1]
                for i in bs_idxs:
                    if i == indices[k]:
                        continue
                    if maxfeval is not None and self.func_evals >= maxfeval:
                        break
                    neighbor.bitstring[indices[k]] = 0 # Set old one to 0
                    neighbor.bitstring[i] = 1 # set new one to 1
                    if neighbor.bitstring in self.tabu_list:
                        # Skip this neighbour
                        neighbor.bitstring[i] = 0 # set new one to 0
                        neighbor.bitstring[indices[k]] = 1 # Set old one back to 1
                        continue

                    #Non tabu neighbour, get fitness
                    bsstr = neighbor.bitstring.to01()
                    if bsstr in self.visited_cache:
                        neighbor.fitness = self.visited_cache[bsstr]
                    else:
                        neighbor.fitness = self.ffunc(neighbor.bitstring)
                        self.visited_cache[bsstr] = neighbor.fitness
                        self.func_evals += 1
                        self.cumulative_fit += neighbor.fitness

                    # If we improved, immediately return
                    if self.minmax*neighbor.fitness > self.minmax*self.current_candidate.fitness:
                        self.current_candidate = copy.deepcopy(neighbor)
                        self.tabu_list.append(neighbor.bitstring)
                        if self.best_candidate is None or self.best_candidate.fitness*self.minmax < self.current_candidate.fitness*self.minmax:
                            self.best_candidate = copy.deepcopy(self.current_candidate)
                        return

                    # Update best neighbour
                    if best_neighbor is None:
                        best_neighbor = copy.deepcopy(neighbor)
                    elif self.minmax*neighbor.fitness > self.minmax*best_neighbor.fitness:
                        best_neighbor = copy.deepcopy(neighbor)

                    neighbor.bitstring[i] = 0 # set new one to 0
                    neighbor.bitstring[indices[k]] = 1 # Set old one back to 1
                    neighbor.fitness = self.current_candidate.fitness

        if best_neighbor is None:
            # If the entire neighbourhood is tabu, return random point
            best_neighbor = individual(self.bs_size, boundary_list=self.boundary_list)
            bsstr = best_neighbor.bitstring.to01()
            if bsstr in self.visited_cache:
                best_neighbor.fitness = self.visited_cache[bsstr]
            else:
                best_neighbor.fitness = self.ffunc(best_neighbor.bitstring)
                self.visited_cache[bsstr] = best_neighbor.fitness
                self.func_evals += 1
                self.cumulative_fit += best_neighbor.fitness

        # We always set the best neighbour as current candidate, even if its worse
        self.tabu_list.append(best_neighbor.bitstring)
        self.current_candidate = best_neighbor
        if self.best_candidate is None or self.best_candidate.fitness*self.minmax < self.current_candidate.fitness*self.minmax:
            self.best_candidate = copy.deepcopy(self.current_candidate)

#TODO: Make OrderedGreedyTabu?
