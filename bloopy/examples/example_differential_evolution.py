import numpy as np
import sys
from timeit import default_timer as timer
import random

import fitness_functions as ff
import dynamic_programming as dp
import bloopy.algorithms.differential_evolution as de
import bloopy.utils as utils

## Generate a (randomized) MK fitness function
k = 4;
m = 33*(k-1);
randomMK = True
if randomMK:
    mk_func = ff.random_MK_function(m, k)
    mk_func.generate()
else:
    mk_func = ff.adjacent_MK_function(m, k)
    mk_func.generate()

## Find optimal solution using dynamic programming for comparison
best_dp_fit = dp.dp_solve_MK(mk_func)
print("Max fitness DP:", best_dp_fit)

# (We also have bruteforce solves but it is exponentially slow.
# Only use it for bitstrings of sizes < 20 to check.
#best_fit, sol = dp.bruteforce_MK_solve(mk_func)
#print("Max fitness bruteforce:", best_fit)

ffunc = mk_func.get_fitness
bitstring_size = m

# Continuous algorithms require a search space to operate
# NOTE: Continuous algorithms can be applied to low dimensional discrete
#   problems with many values per parameter. Bitstring based problems, i.e.
#   only 2 values per dimension are poorly suited.
searchspace = utils.create_bitstring_searchspace(m)
converter = utils.bitstring_as_discrete(searchspace, mk_func.get_fitness)
fitness_func = converter.get_fitness

# Define the Basin Hopping algorithm

count = 1
for vals in searchspace.values():
    count *= len(vals)
print("Points in searchspace:", count)

## Run differential evolution
method = "best1bin"
popsize = 150
recomb = 0.7
mutate = (0.2, 0.7)

minvar = 0.1
maxf = 10000
iterations = int(maxf/(popsize * m) - 1)

test_diffevo = de.differential_evolution(fitness_func,
        1,
        searchspace,
        method=method,
        mutation=mutate,
        recombination=recomb,
        hillclimb=False,#For accurate feval measurements
        pop_size=popsize)

x = test_diffevo.solve(min_variance=minvar,
            max_iter=iterations,
            max_time=30,#seconds
            stopping_fitness=0.98*best_dp_fit,
            max_funcevals=maxf)

print("Best fitness:",x[0],", fraction of optimal {0:.4f}".format(x[0]/float(best_dp_fit)))
print("Function evaluations:", x[2])
