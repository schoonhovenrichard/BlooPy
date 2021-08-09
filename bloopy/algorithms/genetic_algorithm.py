import random
from timeit import default_timer as timer
from bitarray import bitarray
import numpy as np

from bloopy.individual import individual
import bloopy.utils as utils

r"""
NOTE: Custom GAs can be defined by supplying different components for
    selection, reproduction, and fitness. If the user wants to change
    the mutation operator (currently point mutation), one can create
    a child class and override the 'point_mutate' method.
"""

class genetic_algorithm:
    def __init__(self,
            fitness_function,
            reproductor,
            selector,
            population_size,
            bitstring_size,
            min_max_problem=1, #1 for maximization problems, -1 for minimization
            searchspace=None,
            input_pop=None,
            mutation=0.05):
        r"""
        Base genetic algorithm. Most functionalities can be adapted
            by changing input component functions. Only mutation is
            a class method which can be changed by creating a child
            class and overriding the 'point_mutate' method.

        Args:
            fitness_function (bitarray() -> float): Function that
                scores fitness of bitstrings.
            reproductor: Function that creates offspring solutions.
            selector: Function that selects fittest individuals.
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
            mutation (float): (optional) Parameter determines how
                many mutations are applied to each individual each
                generation. The parameter is multiplied times the
                bitstring length to get the nr of point mutatations.
        """
        self.ffunc = fitness_function
        self.reproductor = reproductor
        self.selector = selector
        self.pop_size = population_size
        self.bs_size = bitstring_size
        self.minmax = min_max_problem
        if searchspace is None:
            self.boundary_list = None
        else:
            self.boundary_list = utils.generate_boundary_list(searchspace)
        self.visited_cache = dict()
        self.func_evals = 0
        if mutation is not None:
            self.mutation = max(1, int(self.bs_size * mutation))
        else:
            self.mutation = None

        if input_pop is None:
            print("No input population provided, generating random initial population...")
            self.generate_random_pop()
        else:
            self.current_pop = input_pop
        assert self.pop_size == len(self.current_pop)
        
        self.set_fit_pop(self.current_pop)

    def set_fit_pop(self, pop):
        r"""
        Set fitness of the current population.
        """
        for i in range(len(pop)):
            bsstr = pop[i].bitstring.to01()
            if bsstr in self.visited_cache:
                pop[i].fitness = self.visited_cache[bsstr]
            else:
                pop[i].fitness = self.ffunc(pop[i].bitstring)
                self.visited_cache[bsstr] = pop[i].fitness
                self.func_evals += 1

    def generate_random_pop(self):
        r"""
        Generate a randomly initialized population.
        """
        self.current_pop = []
        for i in range(self.pop_size):
            new_specimen = individual(self.bs_size, boundary_list=self.boundary_list)
            bsstr = new_specimen.bitstring.to01()
            if bsstr in self.visited_cache:
                new_specimen.fitness = self.visited_cache[bsstr]
            else:
                new_specimen.fitness = self.ffunc(new_specimen.bitstring)
                self.func_evals += 1
                self.visited_cache[bsstr] = new_specimen.fitness
            self.current_pop.append(new_specimen)

    def current_best(self):
        r"""
        Return current best solution in the population.
        """
        best_sol = self.current_pop[0]
        for i in range(1, self.pop_size):
            if self.minmax * best_sol.fitness < self.minmax * self.current_pop[i].fitness:
                best_sol = self.current_pop[i]
        return best_sol

    def get_fitnesses(self):
        r"""
        Return fitness list for all individuals in population.
        """
        flist = []
        for k in range(self.pop_size):
            flist.append(self.current_pop[k].fitness)
        return flist

    def create_offspring(self, parents):
        r"""
        Create offspring solutions.
        """
        children = self.reproductor(parents)
        return children

    def point_mutate(self, indiv):
        r"""
        Perform a random point mutation on each individual.
        If self.boundary_list is not None, chooses a random segment,
            and selects a random element from it.
        """
        if self.boundary_list is None:
            pos = random.randint(0, len(indiv.bitstring)-1)
            indiv.bitstring[pos] = not indiv.bitstring[pos]
            indiv.fitness = None
        else:
            indices = [i for i, x in enumerate(list(indiv.bitstring)) if x]
            substr = random.randint(0, len(self.boundary_list)-1)
            indiv.bitstring[indices[substr]] = 0
            pos = random.randint(self.boundary_list[substr][0], self.boundary_list[substr][1])
            indiv.bitstring[pos] = 1
            indiv.fitness = None

    def one_generation(self):
        r"""
        Run the GA for one generation.
        """
        parents = self.current_pop

        # Reproductive step
        children = self.create_offspring(parents)

        # Mutation step
        for child in children:
            for mut in range(self.mutation):
                self.point_mutate(child)
        self.set_fit_pop(children)

        # Selection step
        self.current_pop = self.selector(parents, children, self.minmax)

    def compute_variance(self):
        r"""
        Compute variance of the current population.
        """
        variance = 0.0
        fitnesses = self.get_fitnesses()
        best_fit = self.current_best().fitness
        for i in range(self.pop_size):
            variance += (fitnesses[i] - best_fit)**2
        variance = variance / float(self.pop_size)
        return variance

    def get_bitstrings_pop(self):
        bss = []
        for i in range(self.pop_size):
            bss.append(self.current_pop[i].bitstring)
        return bss

    def solve(self,
            min_variance,
            max_iter,
            no_improve,
            max_time,
            stopping_fitness,
            max_funcevals=None,
            verbose=True
            ):
        r"""
        Solve problem using the algorithm until certain conditions are met.

        Args:
            min_variance (float): Stop solving if variance below threshold.
            max_iter (int): Maximum number of generations to run.
            no_improve (int): Terminate if no improvement found for
                this many generations.
            max_time (int): Max running time in seconds.
            stopping_fitness (float): Stop evaluation if this fitness is
                reached. If we do not know, put +-np.inf.
            max_funcevals (int): (optional) Maximum number of fitness
                function evaluations before terminating.
            verbose (bool): (optional) Run with the GA continually
                printing status updates. Default is True.

        Returns tuple of:
            best_fit (float): Best fitness reached.
            self.current_best (individual): Best individual found.
            best_gen_so_far (int): Generation when best was found.
            variance (float):  Termination variance of population.
            self.func_evals (int): Total number of Fevals performed.
        """
        generation = 0
        best_fit = self.current_best().fitness
        nonterminate = True
        best_gen_so_far = 0
        variance = self.compute_variance()
        self.maxfeval = max_funcevals

        begintime = timer()
        while generation < max_iter and nonterminate:
            if max_funcevals is not None and self.func_evals >= max_funcevals:
                if verbose:
                    print("Max fitness evaluations reached of {0} in {1} generations, terminating...".format(self.func_evals, generation))
                break

            end = timer()
            elapsed_time = end - begintime
            if verbose:
                print('Running generation {0} |Elapsed (s): {1:.2f} |Best fit: {2:.1f} |Variance: {3:.4f}|\r'.format(generation, elapsed_time, best_fit, variance), end="")

            self.one_generation()

            generation += 1
            current_fit = self.current_best().fitness
            if self.minmax * best_fit < self.minmax * current_fit:
                best_gen_so_far = generation
                best_fit = current_fit

            if stopping_fitness is not None:
                if self.minmax * best_fit >= self.minmax * stopping_fitness:
                    nonterminate = False
                    if verbose:
                        print("Stopping fitness reached, terminating...")

            variance = self.compute_variance()
            if variance < min_variance:
                nonterminate = False
                if verbose:
                    print("Minimal variance reached, terminating...")

            end = timer()
            elapsed_time = end - begintime 
            if elapsed_time > max_time:
                nonterminate = False
                if verbose:
                    print("Max running time reached, terminating...")

            if generation - best_gen_so_far > no_improve:
                nonterminate = False
                if verbose:
                    print("No improvement found for {0} generations, terminating...".format(no_improve))

        if verbose:
            print("Terminated after {0} generations with best fitness: {1:.3f} | # of fitness evals: {2}".format(generation, best_fit, self.func_evals))
        return (best_fit, self.current_best(), best_gen_so_far, variance, self.func_evals)
