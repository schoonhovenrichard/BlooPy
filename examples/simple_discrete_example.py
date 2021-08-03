import numpy as np
import itertools as it

import algorithms.local_search as mls
import utils

### Construct some categorical discrete space
searchspace = {"x1": [1,2,3,4,5,6],
               "x2": ["foo", "bar"],
               "x3": [16, 32, 64, 128],
               "x4": ["a", "b", "c", "d", "e"]}
ssvalues = list(searchspace.values())

### Give all possible (x1,x2,x3,x4) a random fitness value
var_names = sorted(searchspace)
possible_xs = list(it.product(*(searchspace[key] for key in var_names)))
print("Size of search space:", len(possible_xs))

# Define fitness function
fitness_values = np.arange(1, 1 + len(possible_xs))
np.random.shuffle(fitness_values)

# Calculate bitstring size
boundary_list = utils.generate_boundary_list(searchspace)
bsize = utils.calculate_bitstring_length(searchspace)
print("Size of bitstring:", bsize)

def map_listvariable_to_index(vec):
    r"""For discrete categorical problems, bitstrings are implemented
      as segments where one bit is active in each segment, and this bit
      designates the parameter value for that variable."""
    # Find indices of active bits:
    indices = []
    it = 0
    for j, var in enumerate(vec):
        vals = ssvalues[j]
        for k, x in enumerate(vals):
            if x == var:
                indices.append(k+it)
                break
    multip = len(possible_xs)
    index = 0
    for i, key in enumerate(searchspace.keys()):
        add = indices[i]
        multip /= len(searchspace[key])
        add *= multip
        index += add
    return int(index)

# Map each entry to a unique index, which points to a random fitness value
fitness_func = lambda vec: fitness_values[map_listvariable_to_index(vec)]

# Create discrete space class
disc_space = utils.discrete_space(fitness_func, searchspace)



### Configure Local Search algorithm (RandomGreedy MLS in this case)
iterations = 10000 # Max number of random restarts
minmax = -1 # -1 for minimization problem, +1 for maximization problem
if minmax == 1:
    optfit = len(possible_xs)
elif minmax == -1:
    optfit = 1
maxfeval = 100000 # Number of unique fitness queries MLS is allowed

test_mls = mls.RandomGreedyMLS(disc_space.fitness,
        bsize,
        minmax,
        searchspace=searchspace)

best_fit, _, fevals = test_mls.solve(iterations,
            max_time=10,#seconds
            stopping_fitness=optfit,#1 is optimal value so we can stop
            max_funcevals=maxfeval,
            verbose=False)
#            verbose=True)
print("Best fitness found:", best_fit, "in", fevals, "evaluations | optimal fitness:", optfit)
