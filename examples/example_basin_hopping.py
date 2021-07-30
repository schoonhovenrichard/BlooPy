import numpy as np
import sys
from timeit import default_timer as timer
import random

import fitness_functions as ff
import dynamic_programming as dp
import algorithms.basin_hopping as bashop
import algorithms.local_minimize as minim
import utils

## Generate a (randomized) MK fitness function
k = 4;
m = 6*(k-1);
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
print("POINTS IN SEARCHSPACE:", count)

BASH = True
MINIM = False

if BASH:
    ## Run basin hopping
    # supported_methods = ["Nelder-Mead", "Powell", "CG", "L-BFGS-B", "COBYLA", "SLSQP", "BFGS"]
    method = "SLSQP"
    temperature = 1.0
    iterations = 10000
    test_bash = bashop.basin_hopping(fitness_func,
            1,
            searchspace,
            T=temperature,
            method=method)

    iterations = 10000
    x = test_bash.solve(max_iter=iterations,
                max_time=10,#seconds
                stopping_fitness=0.98*best_dp_fit,
                max_funcevals=10000)

    print("Best fitness:",x[0],", fraction of optimal {0:.4f}".format(x[0]/float(best_dp_fit)))
    print("Function evaluations:", x[2])

if MINIM:
    ## Run local minimization
    # supported_methods = ["Nelder-Mead", "Powell", "CG", "L-BFGS-B", "COBYLA", "SLSQP", "BFGS"]
    method = "CG"
    test_minim = minim.local_minimizer(fitness_func,
            1,
            searchspace,
            method=method)

    iterations = 10000
    x = test_minim.solve(max_iter=iterations,
                max_time=10,#seconds
                stopping_fitness=0.98*best_dp_fit,
                max_funcevals=10000)

    print("Best fitness:",x[0],", fraction of optimal {0:.4f}".format(x[0]/float(best_dp_fit)))
    print("Function evaluations:", x[2])
