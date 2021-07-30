import numpy as np
import sys
from timeit import default_timer as timer
import random

import fitness_functions as ff
import dynamic_programming as dp
import mutation_functions as mut
import reproductive_functions as rep
import selection_functions as sel
import algorithms.genetic_local_search as gls
import algorithms.hillclimbers as hill

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
population_size = 30
reproductor = rep.uniform_crossover
selector = sel.select_best_half 
bitstring_size = m
hillclimber = hill.RandomGreedyHillclimb

test_gls = gls.genetic_local_search(fitness_func,
            reproductor,
            selector,
            population_size,
            bitstring_size,
            hillclimber,
            min_max_problem=1,
            input_pop=None)

x = test_gls.solve(min_variance=0.1,
            max_iter=1000,
            no_improve=300,
            max_time=60,#seconds
            stopping_fitness=0.98*best_dp_fit,
            max_funcevals=200000)
print("Best fitness:",x[0],", fraction of optimal {0:.4f}".format(x[0]/float(best_dp_fit)))
