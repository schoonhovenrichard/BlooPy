import numpy as np
from bitarray.util import ba2int
import algorithms.local_search as mls

# Define fitness function and searchspace
N = 20 # size of bitstring

fitness_values = np.arange(1, 1 + 2**N)
np.random.shuffle(fitness_values)
fitness_func = lambda bs: fitness_values[ba2int(bs)]
print("Size of search space:", 2**N)

# Configure Local Search algorithm (RandomGreedy MLS in this case)
iterations = 10000 # Max number of random restarts
minmax = -1 # -1 for minimization problem, +1 for maximization problem
maxfeval = 100000 # Number of unique fitness queries MLS is allowed

test_mls = mls.RandomGreedyMLS(fitness_func,
        N,
        minmax)

best_fit, _, fevals = test_mls.solve(iterations,
            max_time=10,#seconds
            stopping_fitness=1,#1 is optimal value so we can stop
            max_funcevals=maxfeval,
            verbose=True)
print("Best fitness found:", best_fit, "in", fevals, "evaluations | optimal fitness:", 1)
