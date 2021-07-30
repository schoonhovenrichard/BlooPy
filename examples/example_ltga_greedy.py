import numpy as np
import sys
from timeit import default_timer as timer

sys.path.insert(1, '/ufs/schoonho/Documents/PhD/EvoPy')
import fitness_functions as ff
import dynamic_programming as dp
import ltga_greedy as lg
import mutation_functions as mut
import reproductive_functions as rep
import selection_functions as sel

k = 4;
m = 133*(k-1);

adj_mk_func = ff.adjacent_MK_function(m, k)
adj_mk_func.generate()
#rand_mk_func = ff.random_MK_function(m, k)
#rand_mk_func.generate()

best_dp_fit = dp.dp_solve_adjacentMK(adj_mk_func)
print("Max fitness DP:", best_dp_fit)

fitness_func = adj_mk_func.get_fitness
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
            max_time=3600,#seconds
            stopping_fitness=0.95*best_dp_fit,
            max_funcevals=100000)
print("\nBest fitness:",x[0],", fraction of optimal {0:.4f}".format(x[0]/float(best_dp_fit)))
