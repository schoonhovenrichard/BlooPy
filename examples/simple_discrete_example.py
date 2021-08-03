import numpy as np
import itertools as it

import algorithms.local_search as mls
import utils

class categorical_fitness:
    def __init__(self, sspace):
        self.sspace = sspace
        self.ssvalues = list(self.sspace.values())#Shorthand

        ### Give all possible (x1,x2,x3,x4) a random fitness value
        var_names = sorted(self.sspace)
        self.possible_xs = list(it.product(*(sspace[key] for key in var_names)))
        print("Size of search space:", len(self.possible_xs))

        # Define fitness function
        self.fitness_values = np.arange(1, 1 + len(self.possible_xs))
        np.random.shuffle(self.fitness_values)

        # Calculate bitstring size
        self.bsize = utils.calculate_bitstring_length(self.sspace)
        print("Size of bitstring:", self.bsize)

    def map_listvariable_to_index(self, vec):
        r"""For discrete categorical problems, bitstrings are implemented
          as segments where one bit is active in each segment, and this bit
          designates the parameter value for that variable."""
        # This function looks complicated, but it merely uniquely maps each
        #  possible vector to an index to get a random fitness value.
        indices = []
        it = 0
        for j, var in enumerate(vec):
            vals = self.ssvalues[j]
            for k, x in enumerate(vals):
                if x == var:
                    indices.append(k+it)
                    break
        multip = len(self.possible_xs)
        index = 0
        for i, key in enumerate(self.sspace.keys()):
            add = indices[i]
            multip /= len(self.sspace[key])
            add *= multip
            index += add
        return int(index)

    def fitness(self, vec):
        # Map each entry to a unique index, which points to a random fitness value
        return self.fitness_values[self.map_listvariable_to_index(vec)]


### Construct some categorical discrete space
searchspace = {"x1": [1,2,3,4,5,6],
               "x2": ["foo", "bar"],
               "x3": [16, 32, 64, 128],
               "x4": ["a", "b", "c", "d", "e"]}

categorical_fit = categorical_fitness(searchspace)

# Create discrete space class
disc_space = utils.discrete_space(categorical_fit.fitness, searchspace)


### Configure Local Search algorithm (RandomGreedy MLS in this case)
iterations = 10000 # Max number of random restarts
minmax = -1 # -1 for minimization problem, +1 for maximization problem
if minmax == 1:
    optfit = len(categorical_fit.possible_xs)
elif minmax == -1:
    optfit = 1
maxfeval = 100000 # Number of unique fitness queries MLS is allowed

test_mls = mls.RandomGreedyMLS(disc_space.fitness,
        categorical_fit.bsize,
        minmax,
        searchspace=searchspace)

best_fit, _, fevals = test_mls.solve(iterations,
            max_time=10,#seconds
            stopping_fitness=optfit,#1 is optimal value so we can stop
            max_funcevals=maxfeval,
            verbose=True)
print("Best fitness found:", best_fit, "in", fevals, "evaluations | optimal fitness:", optfit)
