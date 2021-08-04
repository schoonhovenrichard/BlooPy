import numpy as np
from timeit import default_timer as timer
import random

import fitness_functions as ff
import dynamic_programming as dp
import bloopy.algorithms.local_search as mls
import bloopy.algorithms.iterative_local_search as ils

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
iterations = 1000
bitstring_size = m

MLS = False
ILS = True

if MLS:
    MLS_TYPE = "RandomGreedy"
    #MLS_TYPE = "Best"
    #MLS_TYPE = "OrderedGreedy"
    #MLS_TYPE = "Stochastic"

    restart = False
    if MLS_TYPE == "RandomGreedy":
        test_mls = mls.RandomGreedyMLS(fitness_func,
                bitstring_size,
                1,
                restart_search=restart)
    elif MLS_TYPE == "OrderedGreedy":
        test_mls = mls.OrderedGreedyMLS(fitness_func,
                bitstring_size,
                1,
                restart_search=restart,
                order=None)
    elif MLS_TYPE == "Best":
        test_mls = mls.BestMLS(fitness_func,
                bitstring_size,
                1)
    elif MLS_TYPE == "Stochastic":
        test_mls = mls.StochasticMLS(fitness_func,
                bitstring_size,
                1)

    x = test_mls.solve(iterations,
                max_time=60,#seconds
                stopping_fitness=0.98*best_dp_fit,
                max_funcevals=200000,
                verbose=True)
    print("Best fitness:",x[0],", fraction of optimal {0:.4f}".format(x[0]/float(best_dp_fit)))

if ILS:
    ILS_TYPE = "RandomGreedy"
    #ILS_TYPE = "Best"
    #ILS_TYPE = "OrderedGreedy"
    #ILS_TYPE = "Stochastic"

    restart = True
    walksize = 50
    no_improve = 100
    if ILS_TYPE == "RandomGreedy":
        test_ils = ils.RandomGreedyILS(fitness_func,
                bitstring_size,
                1,
                walksize,
                noimprove=no_improve,
                restart_search=restart)
    elif ILS_TYPE == "OrderedGreedy":
        test_ils = ils.OrderedGreedyILS(fitness_func,
                bitstring_size,
                1,
                walksize,
                noimprove=no_improve,
                restart_search=restart,
                order=None)
    elif ILS_TYPE == "Best":
        test_ils = ils.BestILS(fitness_func,
                bitstring_size,
                1,
                walksize,
                noimprove=no_improve)
    elif ILS_TYPE == "Stochastic":
        test_ils = ils.StochasticILS(fitness_func,
                bitstring_size,
                1,
                walksize,
                noimprove=no_improve)

    x = test_ils.solve(iterations,
                max_time=60,#seconds
                stopping_fitness=best_dp_fit,
                max_funcevals=200000,
                verbose=True)
    print("Best fitness:",x[0],", fraction of optimal {0:.4f}".format(x[0]/float(best_dp_fit)))
