import numpy as np
import sys
from timeit import default_timer as timer

import fitness_functions as ff
import dynamic_programming as dp
import algorithms.ltga_greedy as lg
import mutation_functions as mut
import reproductive_functions as rep
import selection_functions as sel

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
population_size = 200
bitstring_size = m
test_ltga = lg.ltga_greedy(fitness_func,
            population_size,
            bitstring_size,
            min_max_problem=1,
            input_pop=None,
            maxdepth=8)

x = test_ltga.solve(min_variance=0.1,
            max_iter=1000,
            no_improve=300,
            max_time=30,#seconds
            stopping_fitness=0.98*best_dp_fit,
            max_funcevals=200000)
print("\nBest fitness:",x[0],", fraction of optimal {0:.4f}".format(x[0]/float(best_dp_fit)))
