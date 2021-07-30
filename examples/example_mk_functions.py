import numpy as np
import sys
from timeit import default_timer as timer
import random

sys.path.insert(1, '/ufs/schoonho/Documents/PhD/EvoPy')
import fitness_functions as ff
import dynamic_programming as dp

random.seed(123)

k = 4;
m = 4*(k-1);

adj_mk_func = ff.adjacent_MK_function(m, k)
#adj_mk_func = ff.adjacent_MK_function_str(m, k)
rand_mk_func = ff.random_MK_function(m, k)
adj_mk_func.generate()
rand_mk_func.generate()
rand_mk_func.set_random_bitmask()

start1 = timer()
best_fit, sol = dp.bruteforce_MK_solve(adj_mk_func)
#best_fit, sol = dp.bruteforce_MK_solve_str(adj_mk_func)
end1 = timer()
print("Bruteforce evaluation time:", 1000*(end1-start1))
print("Max fitness bruteforce:", best_fit)
#print("Best solution:", sol)

start2 = timer()
best_dp_fit = dp.dp_solve_adjacentMK(adj_mk_func)
#best_dp_fit = dp.dp_solve_adjacentMK_str(adj_mk_func)
end2 = timer()
print("DP evaluation time:", 1000*(end2-start2))
print("Max fitness DP:", best_dp_fit)
