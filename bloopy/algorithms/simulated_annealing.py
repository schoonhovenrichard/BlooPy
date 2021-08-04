import random
import sys
from timeit import default_timer as timer
import copy
import numpy as np
from bitarray import bitarray

import bloopy.utils as utils
from bloopy.individual import individual
from bloopy.algorithms.local_search import multi_start_local_search_base

class simulated_annealing(multi_start_local_search_base):
    def __init__(self, fitness_function, bitstring_size, min_max_problem, exploration, T0=10000000.0, Tmin=1e-5, hillclimb=None, searchspace=None, neighbour="Hamming"):
        r"""
        Base Simulated Annealing algorithm. Different options for
            temperature distribution are available.

        Args:
            fitness_function (bitarray() -> float): Function that
                scores fitness of bitstrings.
            bitstring_size (int): Length of the bitstring instances.
            min_max_problem (int): 1 if maximization problem, -1 for
                    minimization problem. Default is 1.
            exploration (int): Defines basin of exploration, i.e.
                number of point mutations performed to generate new.
            T0 (float): Starting temperature.
            Tmin (float): Termination temperature.
            hillclimb (func): Optional hillclimber function.
            searchspace (dict): Mapping of settings to fitnesses
            neighbour (string): Method for generating neighbour solutions to visit.
        """
        super().__init__(fitness_function, bitstring_size, min_max_problem, searchspace=searchspace, neighbour=neighbour)
        self.exploration = exploration
        self.T0 = T0
        self.Tmin = Tmin
        self.hillclimb = hillclimb
        self.C = -np.log(self.T0/float(self.Tmin))

    def probability(self, E1, E2, t):
        r"""
        Returns acceptance probability for two fitness energies.
        """
        return 1 if self.minmax*E1 > self.minmax*E2 else np.exp(self.minmax*(E1-E2)/float(t))

    def temperature(self, k):
        r"""
        Regulates temperature depending on iterations k.
        """
        #return self.T0 * (1.0 / float(k) - 1.0 / float(self.iters + 1))
        #return self.T0 / float(k)
        #return self.T0 / float(np.log(k+1))
        return self.T0 * np.exp(self.C*k/float(self.iters))
    
    def generate_initial_candidate(self):
        r"""
        Generate starting solution.
        """
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

    def generate_new_candidate(self, maxfeval):
        r"""
        Generate a new current candidate.
        """
        new_candidate = individual(self.current_candidate.size, bitstring=copy.deepcopy(self.current_candidate.bitstring), boundary_list=self.boundary_list)
        new_candidate.fitness = self.current_candidate.fitness

        # Generate random new candidate
        for i in range(self.exploration):
            self.point_mutate(new_candidate)

        # Get fitness of new candidate
        bsstr = new_candidate.bitstring.to01()
        if bsstr in self.visited_cache:
            new_candidate.fitness = self.visited_cache[bsstr]
        else:
            new_candidate.fitness = self.ffunc(new_candidate.bitstring)
            self.visited_cache[bsstr] = new_candidate.fitness
            self.func_evals += 1
            self.cumulative_fit += new_candidate.fitness
        if self.best_candidate is None or self.best_candidate.fitness*self.minmax < new_candidate.fitness*self.minmax:
            self.best_candidate = copy.deepcopy(new_candidate)

        # Hillclimb candidate if specified
        if self.hillclimb is not None:
            new_candidate, extra_fevals, totfit, self.visited_cache = self.hillclimb(new_candidate, self.ffunc, self.minmax, self.func_evals, maxfeval, self.visited_cache, self.nbour_method)
            self.func_evals += extra_fevals
            self.cumulative_fit += totfit
            if self.best_candidate is None or self.best_candidate.fitness*self.minmax < new_candidate.fitness*self.minmax:
                self.best_candidate = copy.deepcopy(new_candidate)
        return new_candidate

    def solve(self, iters, max_time, stopping_fitness, max_funcevals=None, verbose=True):
        r"""
        Solve problem using the algorithm until certain conditions are met.

        Args:
            max_time (int): Max running time in seconds.
            stopping_fitness (float): Stop evaluation if this fitness is
                reached. If we do not know, put +-np.inf.
            max_funcevals (int): (optional) Maximum number of fitness
                function evaluations before terminating.
            verbose (bool): (optional) Run with the GA continually
                printing status updates. Default is True.

        Returns tuple of:
            best_fit (float): Best fitness reached.
            self.best_candidate (individual): Best individual found.
            self.func_evals (int): Total number of Fevals performed.
            self.cumulative_fit (float): Cumulative fitness of all
                fitness function evaluations.
        """
        self.iters = iters
        nonterminate = True
        self.generate_initial_candidate()
        begintime = timer()
        accept, reject, improve = 0, 0, 0
        for k in range(1, self.iters + 1):
            if not nonterminate:
                break
            temperature = self.temperature(k)

            ## If temperature is too low, restart
            if temperature < self.Tmin:
                print("Restarting...","\n")
                self.solve(max_time, stopping_fitness, max_funcevals, verbose)
                break

            new = self.generate_new_candidate(max_funcevals)
            E1 = self.minmax * self.current_candidate.fitness
            E2 = self.minmax * new.fitness

            p = self.probability(E1, E2, temperature)
            q = random.random()
            if p > q:
                accept += 1
                self.current_candidate = new
                if self.best_candidate.fitness*self.minmax < self.current_candidate.fitness*self.minmax:
                    improve += 1
                    self.best_candidate = self.current_candidate
            else:
                reject += 1

            best_fit = self.best_candidate.fitness
            end = timer()
            elapsed_time = end - begintime
            if verbose:
                print('Running iteration {0} |Elapsed (s): {1:.2f} |Best fit: {2:.1f} |Temp {3:.8f} |A/R/I {4},{5},{6} |\r'.format(k, elapsed_time, best_fit, temperature, accept, reject, improve), end="")

            if stopping_fitness is not None:
                if self.minmax * best_fit >= self.minmax * stopping_fitness:
                    nonterminate = False
                    if verbose:
                        print("\nStopping fitness reached, terminating...")

            end = timer()
            elapsed_time = end - begintime
            if elapsed_time > max_time:
                nonterminate = False
                if verbose:
                    print("\nMax running time reached, terminating...")

            if max_funcevals is not None and self.func_evals >= max_funcevals:
                nonterminate = False
                if verbose:
                    print("\nBest fitness evaluations reached of {0} in {1} iterations, terminating...".format(self.func_evals, k))
            it = k

        if verbose:
            print("Terminated after {0} iterations with best fitness: {1:.3f} | # of fitness evals: {2}".format(it, best_fit, self.func_evals))
        return (best_fit, self.best_candidate, self.func_evals, self.cumulative_fit)

class simulated_annealing_KernelTuner(multi_start_local_search_base):
    def __init__(self, iterations, fitness_function, bitstring_size, min_max_problem, T0=1.0, Tmin=0.00001, destroy_prob=0.2, alpha=0.99, hillclimb=False, searchspace=None):
    #def __init__(self, iterations, fitness_function, bitstring_size, min_max_problem, T0=1.0, Tmin=0.001, destroy_prob=0.2, alpha=0.95, hillclimb=False, searchspace=None):
        r"""
        Base Simulated Annealing algorithm. Different options for
            temperature distribution are available.

        Args:
            iterations (int): Number of random restarts.
            fitness_function (bitarray() -> float): Function that
                scores fitness of bitstrings.
            bitstring_size (int): Length of the bitstring instances.
            min_max_problem (int): 1 if maximization problem, -1 for
                    minimization problem. Default is 1.
            exploration (int): Defines basin of exploration, i.e.
                number of point mutations performed to generate new.
            T0 (float): Starting temperature.
            Tmin (float): Termination temperature.
            hillclimb (bool): Shoul current candidates be hillclimbed,
                default is True.
            boundary_list (list(tuple(int))): (optional) None if 
                regular bitstrings. Otherwise, list of tuples 
                (start, end) of each segment of the bitstring in
                which we can have only one 1 that points to the
                element of the list that is active.
        """
        super().__init__(iterations, fitness_function, bitstring_size, min_max_problem, searchspace=searchspace)
        self.T0 = T0
        self.Tmin = Tmin
        self.hillclimb = hillclimb
        self.destroy_prob = destroy_prob
        self.alpha = alpha
        self.accept = 0
        self.reject = 0
        self.improve = 0
        raise Exception("Deprecated")

#    def probability(self, E1, E2, t):
#        r"""
#        Returns acceptance probability for two fitness energies.
#        """
#        print(np.exp((E1-E2)/float(t)))
#        return 1 if E1 > E2 else np.exp(self.minmax*(E1-E2)/float(t))

    def probability(self, old_cost, new_cost, T):
        """annealing equation, with modifications to work towards a lower value"""
        #NOTE these returns are purely for kernel tuning
        # if start pos is not valid, always move
        if old_cost >= 1e20:
            return 1.0
        # if we have found a valid ps before, never move to nonvalid pos
        if new_cost >= 1e20:
            return 0.0

        # always move if new cost is better
        if self.minmax*new_cost > self.minmax*old_cost:
            return 1.0
        # maybe move if old cost is better than new cost depending on T and random value
        return np.exp(self.minmax*(new_cost-old_cost)/T)

    def temperature(self, k):
        r"""
        Regulates temperature depending on iterations k.
        """
        return self.T0 * (self.alpha**k) # Kernel Tuner has this version
    
    def generate_initial_candidate(self):
        r"""
        Generate starting solution.
        """
        self.current_candidate = individual(self.bs_size, boundary_list=self.boundary_list)
        bsstr = self.current_candidate.bitstring.to01()
        if bsstr in self.visited_cache:
            self.current_candidate.fitness = self.visited_cache[bsstr]
        else:
            self.current_candidate.fitness = self.ffunc(self.current_candidate.bitstring)
            self.visited_cache[bsstr] = self.current_candidate.fitness
            self.func_evals += 1
            self.cumulative_fit += self.current_candidate.fitness
        if self.best_candidate is None:
            self.best_candidate = self.current_candidate

    def generate_new_candidate(self, maxfeval):
        r"""
        Generate a new current candidate in the way Ben does.
        """
        new_candidate = individual(self.current_candidate.size, bitstring=copy.deepcopy(self.current_candidate.bitstring), boundary_list=self.boundary_list)
        new_candidate.fitness = self.current_candidate.fitness
        if self.current_candidate.boundary_list is None:
            raise Exception("Not implemented for general bitstrings")
        indices = [i for i, x in enumerate(list(new_candidate.bitstring)) if x]
        for i in range(len(indices)):
            pos = indices[i]
            segment = new_candidate.boundary_list[i]
            if random.random() < self.destroy_prob: # Replace with random value from segment
                new_val = random.randint(segment[0], segment[1])
            else: # Go to adjacent value
                if random.random() > 0.5:
                    new_val = pos + 1
                    if new_val > segment[1]:
                        new_val = segment[0]
                else:
                    new_val = pos - 1
                    if new_val < segment[0]:
                        new_val = segment[1]
            if new_val < segment[0] or new_val > segment[1]:
                print(segment, new_val, pos, indices)
                raise Exception("Out of bounds bitstring segment!")
            new_candidate.bitstring[pos] = 0
            new_candidate.bitstring[new_val] = 1
        bsstr = new_candidate.bitstring.to01()
        if bsstr in self.visited_cache:
            new_candidate.fitness = self.visited_cache[bsstr]
        else:
            new_candidate.fitness = self.ffunc(new_candidate.bitstring)
            self.visited_cache[bsstr] = new_candidate.fitness
            self.func_evals += 1
            self.cumulative_fit += new_candidate.fitness
        if self.hillclimb:
            old_current = individual(self.current_candidate.size, bitstring=copy.deepcopy(self.current_candidate.bitstring), fitness=self.current_candidate.fitness, boundary_list=self.boundary_list)
            self.current_candidate = new_candidate
            self.hillclimb_candidate(maxfeval)
            new_candidate = self.current_candidate
            self.current_candidate = old_current
        return new_candidate

    def solve(self, max_time, stopping_fitness, max_funcevals=None, verbose=True):
        r"""
        Solve problem using the algorithm until certain conditions are met.

        Args:
            max_time (int): Max running time in seconds.
            stopping_fitness (float): Stop evaluation if this fitness is
                reached. If we do not know, put +-np.inf.
            max_funcevals (int): (optional) Maximum number of fitness
                function evaluations before terminating.
            verbose (bool): (optional) Run with the GA continually
                printing status updates. Default is True.

        Returns tuple of:
            best_fit (float): Best fitness reached.
            self.best_candidate (individual): Best individual found.
            self.func_evals (int): Total number of Fevals performed.
            self.cumulative_fit (float): Cumulative fitness of all
                fitness function evaluations.
        """
        nonterminate = True
        self.generate_initial_candidate()
        begintime = timer()
        for k in range(1, self.iters + 1):
            if not nonterminate:
                break
            temperature = self.temperature(k)

            ## If temperature is too low, restart
            if temperature < self.Tmin:
                print("Restarting...","\n")
                self.solve(max_time, stopping_fitness, max_funcevals, verbose)
                break

            new = self.generate_new_candidate(max_funcevals)
            E1 = self.current_candidate.fitness
            E2 = new.fitness

            p = self.probability(E1, E2, temperature)
            q = random.random()
            #if E2 > E1 and E2 < 1e20 and p>0.05:
            #    print(p>q, p,q, E1, E2)
            #    #np.exp((self.minmax*(E2-E1))/temperature))
            if p > q:
                self.accept += 1
                self.current_candidate = new
                if self.best_candidate.fitness*self.minmax < self.current_candidate.fitness*self.minmax:
                    self.improve += 1
                    self.best_candidate = self.current_candidate
            else:
                self.reject += 1

            best_fit = self.best_candidate.fitness
            end = timer()
            elapsed_time = end - begintime
            if verbose:
                print('Running iteration {0} |Elapsed (s): {1:.2f} |Best fit: {2:.1f} |Temp {3:.8f} |A/R/I {4},{5},{6} |\r'.format(k, elapsed_time, best_fit, temperature, self.accept, self.reject, self.improve), end="")

            if stopping_fitness is not None:
                if self.minmax * best_fit >= self.minmax * stopping_fitness:
                    nonterminate = False
                    if verbose:
                        print("\nStopping fitness reached, terminating...")

            end = timer()
            elapsed_time = end - begintime
            if elapsed_time > max_time:
                nonterminate = False
                if verbose:
                    print("\nMax running time reached, terminating...")

            if max_funcevals is not None and self.func_evals >= max_funcevals:
                nonterminate = False
                if verbose:
                    print("\nBest fitness evaluations reached of {0} in {1} iterations, terminating...".format(self.func_evals, k))
            it = k

        if verbose:
            print("Terminated after {0} iterations with best fitness: {1:.3f} | # of fitness evals: {2}".format(it, best_fit, self.func_evals))
        return (best_fit, self.best_candidate, self.func_evals, self.cumulative_fit)
