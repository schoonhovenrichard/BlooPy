import numpy as np
import sys
from timeit import default_timer as timer
import random

sys.path.insert(1, '/ufs/schoonho/Documents/PhD/EvoPy')
import fitness_functions as ff
import dynamic_programming as dp
import genetic_algorithm as ga
import mutation_functions as mut
import reproductive_functions as rep
import selection_functions as sel
import local_search as mls
import iterative_local_search as ils
import simulated_annealing as sa

random.seed(123456)

k = 4;
m = 133*(k-1);

adj_mk_func = ff.adjacent_MK_function(m, k)
#rand_mk_func = ff.random_MK_function(m, k)
adj_mk_func.generate()
#rand_mk_func.generate()
#rand_mk_func.set_random_bitmask()

#start1 = timer()
#best_fit, sol = dp.bruteforce_MK_solve(adj_mk_func)
#end1 = timer()
#print("Bruteforce evaluation time:", 1000*(end1-start1))
#print("Max fitness bruteforce:", best_fit)
#print("Best solution:", sol)

start2 = timer()
best_dp_fit = dp.dp_solve_adjacentMK(adj_mk_func)
end2 = timer()
print("DP evaluation time (ms):", 1000*(end2-start2))
print("Max fitness DP:", best_dp_fit)

fitness_func = adj_mk_func.get_fitness
iterations = 50000
bitstring_size = m
explor = 24
test_sa = sa.simulated_annealing(iterations,
            fitness_func,
            bitstring_size,
            1,
            mut.bs_point_mutate,
            explor)

x = test_sa.solve(max_time=60,#seconds
            stopping_fitness=best_dp_fit,
            max_funcevals=100000,
            verbose=True)
print("Best fitness:",x[0],", fraction of optimal {0:.4f}".format(x[0]/float(best_dp_fit)))
print("Function evals:", x[-1])
