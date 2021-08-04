import random
import sys
from timeit import default_timer as timer
import copy
import numpy as np
import math
from bitarray import bitarray

import bloopy.utils as utils
from bloopy.algorithms.genetic_algorithm import genetic_algorithm
import bloopy.selection_functions as sel
import bloopy.reproductive_functions as rep


class ltga(genetic_algorithm):
    def __init__(self,
            fitness_function,
            population_size,
            bitstring_size,
            min_max_problem=1, #1 for maximization problems, -1 for minimization
            searchspace=None,
            input_pop=None,
            maxdepth=None):
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
        self.sspace = searchspace
        if self.sspace is None:
            self.LOGTOLIN = False
        else:
            self.LOGTOLIN = True
            self.build_loglinear_dicts()
        super().__init__(fitness_function,
                rep.mask_crossover_pair, # The reproductor is fixed now
                None, # Selector is not used
                population_size,
                bitstring_size,
                min_max_problem,
                searchspace,
                input_pop,
                None)#Mutations are not used
        self.boundary_list = None
        if maxdepth is None:
            self.maxdepth = self.bs_size
        else:
            self.maxdepth = maxdepth
        self.single_entropies = [0.0] * self.bs_size
        self.pairwise_entropies = [[0.0]*self.bs_size for i in range(self.bs_size)]
        self.pairwise_info = [[0.0]*self.bs_size for i in range(self.bs_size)]
        self.cluster_info = dict()
        self.linkage_tree = []

    def build_loglinear_dicts(self):
        # Build dicts to go between log and linear bit encoding
        logTOlinear = dict()
        linearTOlog = dict()
        for var in self.sspace:
            logDict = dict()
            linearDict = dict()
            bslen = len(self.sspace[var])
            loglen = int(math.ceil(np.log2(bslen)))
            for i in range(bslen):
                arrlin = np.zeros(shape=(bslen,), dtype=np.bool)
                arrlog = np.zeros(shape=(loglen,), dtype=np.bool)
                arrlin[i] = True
                val = i
                while val > 0:
                    k = int(np.log2(val))
                    arrlog[k] = True
                    val -= 2**k
                bslin = bitarray(arrlin.tolist()).to01()
                bslog = bitarray(arrlog.tolist()).to01()
                if bslin not in linearTOlog:
                    linearDict[bslin] = bslog
                if bslog not in logTOlinear:
                    logDict[bslog] = bslin
            logTOlinear[var] = logDict
            linearTOlog[var] = linearDict
        self.linlog = linearTOlog
        self.loglin = logTOlinear

    def set_fit_pop(self, pop):
        r"""
        Set fitness of the current population.
        """
        for i in range(len(pop)):
            if self.LOGTOLIN:
                self.set_fitness_log(pop[i])
            else:
                self.set_fitness(pop[i])


    def cast_to_linear(self, bstr):
        linstring = ""
        index = 0
        for var in self.sspace:
            currentDict = self.loglin[var]
            bslen = len(self.sspace[var])
            loglen = int(math.ceil(np.log2(bslen)))
            logsegm = bstr[index:index+loglen].to01()
            if logsegm in currentDict:
                linearsegm = currentDict[logsegm]
            else:
                linearsegm = list(currentDict.values())[-1]
            linstring += linearsegm
            index += loglen
        return bitarray(linstring)

    def set_fitness_log(self, sol):
        bsstr = sol.bitstring.to01()
        if bsstr in self.visited_cache:
            # Cache is logarithmic!
            sol.fitness = self.visited_cache[bsstr]
        else:
            #NOTE: sol bitstring is logarithmic
            linbs = self.cast_to_linear(sol.bitstring)
            sol.fitness = self.ffunc(linbs)
            self.visited_cache[bsstr] = sol.fitness
            self.cumulative_fit += sol.fitness
            self.func_evals += 1

    def set_fitness(self, sol):
        bsstr = sol.bitstring.to01()
        if bsstr in self.visited_cache:
            sol.fitness = self.visited_cache[bsstr]
        else:
            sol.fitness = self.ffunc(sol.bitstring)
            self.visited_cache[bsstr] = sol.fitness
            self.cumulative_fit += sol.fitness
            self.func_evals += 1

    def one_generation(self):
        r"""
        Perform one generation of Linkage Tree GA.
        """
        children = self.create_offspring(self.current_pop)
        for child in children:
            self.point_mutate(child)
        self.set_fit_pop(children)
        self.current_pop = children

    def create_offspring(self, parents):
        r"""
        Create offspring solutions. Uses mask crossover and
            mutual information between bits.
        """
        if self.pop_size % 2 != 0:
            raise Exception("Not implemented for uneven numbers!")
        self.build_linkage_tree()
        children = []
        random.shuffle(parents)
        for i in range(self.pop_size//2):
            par1 = parents[2*i]
            par2 = parents[2*i + 1]
            fin_child1 = par1
            fin_child2 = par2
            copy_tree = copy.deepcopy(self.linkage_tree)
            if self.maxdepth == self.bs_size:
                copy_tree.pop() #Root node is pointless as it is all the bits
            while len(copy_tree) > 0:
                crossover_mask = copy_tree.pop()
                child1, child2 = self.reproductor(par1, par2, crossover_mask)
                if child1.bitstring == par1.bitstring and child2.bitstring == par2.bitstring:
                    continue
                
                if self.LOGTOLIN:
                    self.set_fitness_log(child1)
                    self.set_fitness_log(child2)
                else:
                    self.set_fitness(child1)
                    self.set_fitness(child2)

                # Greedy recombination
                if (child1.fitness*self.minmax >= fin_child1.fitness*self.minmax and child1.fitness*self.minmax >= fin_child2.fitness*self.minmax) or (child2.fitness*self.minmax >= fin_child1.fitness*self.minmax and child2.fitness*self.minmax >= fin_child2.fitness*self.minmax):
                    # Update best children
                    fin_child1 = child1
                    fin_child2 = child2
            children += [fin_child1, fin_child2]
        return children

    def compute_single_entropies(self):
        r"""
        Helper function to compute single bit position entropies.
        """
        for bit in range(self.bs_size):
            entropy = 0.0
            count = 0.0
            # Get the number of times that bit is 1 in population.
            for k in range(self.pop_size):
                count += self.current_pop[k].bitstring[bit]
            proportion1s = count / float(self.pop_size)
            if proportion1s <= 0:
                entropy = -1*(1 - proportion1s)*np.log2(1 - proportion1s)
            else:
                if proportion1s >= 1:
                    entropy = -1 * proportion1s * np.log2(proportion1s)
                else:
                    entropy = -1 * proportion1s * np.log2(proportion1s)
                    - (1 - proportion1s) * np.log2(1 - proportion1s)
            self.single_entropies[bit] = np.abs(entropy)

    def compute_pairwise_entropies(self):
        r"""
        Helper function to compute pairwise position entropies.
        """
        arr = np.empty(shape=(self.pop_size, self.bs_size), dtype=np.bool)
        for k in range(self.pop_size):
            bitset = self.current_pop[k].bitstring
            arr[k] = list(bitset)
        for b1 in range(self.bs_size-1):
            entropy = self.single_entropies[b1]
            self.pairwise_entropies[b1][b1] = np.abs(entropy)
            b1row = arr[:,b1]
            b1rows = np.array(len(range(b1+1,self.bs_size))*[b1row])
            b2rows = arr[:,range(b1+1,self.bs_size)]
            count11s = np.all([b1rows, b2rows.transpose()],axis=0).sum(axis=1)
            count00s = np.invert(np.any([b1rows, b2rows.transpose()],axis=0)).sum(axis=1)/float(self.pop_size)
            count10s = (b1rows.sum(axis=1) - count11s)/float(self.pop_size)
            count01s = (b2rows.transpose().sum(axis=1) - count11s)/float(self.pop_size)
            count11s = count11s/float(self.pop_size)
            for b2 in range(b1 + 1, self.bs_size):
                entropy = 0.0
                count00 = count00s[b2-b1-1]
                count01 = count01s[b2-b1-1]
                count10 = count10s[b2-b1-1]
                count11 = count11s[b2-b1-1]
                if count00 != 0:
                    entropy += count00 * np.log2(count00)
                if count10 != 0:
                    entropy += count10 * np.log2(count10)
                if count01 != 0:
                    entropy += count01 * np.log2(count01)
                if count11 != 0:
                    entropy += count11 * np.log2(count11)
                self.pairwise_entropies[b1][b2] = np.abs(entropy)
        entropy = self.single_entropies[self.bs_size-1]
        self.pairwise_entropies[self.bs_size-1][self.bs_size-1] = np.abs(entropy)

    def compute_pairwise_information(self):
        r"""
        Compute pairwise mutual information.
        """
        self.compute_single_entropies()
        self.compute_pairwise_entropies()
        for i in range(self.bs_size):
            for j in range(self.bs_size):
                self.pairwise_info[i][j] = self.single_entropies[i] + self.single_entropies[j] - self.pairwise_entropies[i][j]

    def compute_cluster_information(self, cluster1, cluster2):
        r"""
        Compute mutual information for cluster bit masks.
        """
        if len(cluster1) == 1 and len(cluster2) == 1:
            return self.pairwise_info[cluster1[0]][cluster2[0]]
        cluster_pair = (cluster1, cluster2)
        if cluster_pair in self.cluster_info: # previously computed
            return self.cluster_info[cluster_pair]
        mutual_info = 0.0
        for i in range(len(cluster1)):
            for j in range(len(cluster2)):
                mutual_info += self.pairwise_info[cluster1[i]][cluster2[j]]
        mutual_info = mutual_info / float(len(cluster1) * len(cluster2))
        cluster_pair2 = (cluster2, cluster1)
        self.cluster_info[cluster_pair] = mutual_info
        self.cluster_info[cluster_pair2] = mutual_info
        return mutual_info

    def build_linkage_tree(self, max_depth=None):
        r"""
        Build the linkage tree of bit masks.
        """
        self.cluster_info.clear()
        self.compute_pairwise_information()
        self.linkage_tree = []
        cluster_tracker = []
        last_cluster = []
        for b in range(self.bs_size):
            cluster = tuple([b])
            self.linkage_tree.append(cluster)
            last_cluster = cluster
            cluster_tracker.append(cluster)
        while len(last_cluster) < self.maxdepth:
            best_info = sys.float_info.max
            best_indices = []
            for i in range(len(cluster_tracker)):
                for j in range(i+1, len(cluster_tracker)):
                    distance = self.compute_cluster_information(cluster_tracker[i], cluster_tracker[j])
                    if distance < best_info:
                        best_info = distance
                        best_indices = [i,j]
            v1 = cluster_tracker[best_indices[0]]
            v2 = cluster_tracker[best_indices[1]]
            new_cluster = v1 + v2
            last_cluster = new_cluster
            self.linkage_tree.append(new_cluster)

            # We have merged i and j, so remove them, j is always bigger, remove it first
            del cluster_tracker[best_indices[1]]
            del cluster_tracker[best_indices[0]]
            cluster_tracker.append(new_cluster)
