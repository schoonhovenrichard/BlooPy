import numpy as np
import sys
from timeit import default_timer as timer
import random

import fitness_functions as ff
import dynamic_programming as dp
import algorithms.tabu_search as tabu

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

fitness_func = mk_func.get_fitness
iterations = 100000
bitstring_size = m
tabu_size = 1024

TABU_TYPE = "RandomGreedy"
#TABU_TYPE = "Best"

if TABU_TYPE == "RandomGreedy":
    test_tabu = tabu.RandomGreedyTabu(fitness_func,
            bitstring_size,
            1,
            tabu_size)
elif TABU_TYPE == "Best":
    test_tabu = tabu.BestTabu(fitness_func,
            bitstring_size,
            1,
            tabu_size)

x = test_tabu.solve(iterations,
            max_time=30,#seconds
            stopping_fitness=0.98*best_dp_fit,
            max_funcevals=200000,
            verbose=True)
print("Best fitness:",x[0],", fraction of optimal {0:.4f}".format(x[0]/float(best_dp_fit)), "Function evals:",x[2])
