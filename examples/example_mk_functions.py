import numpy as np
import sys
from timeit import default_timer as timer
import random

import fitness_functions as ff
import dynamic_programming as dp

random.seed(123)

k = 4;
m = 5*(k-1);

randomMK = True
if randomMK:
    mk_func = ff.random_MK_function(m, k)
    mk_func.generate()
else:
    mk_func = ff.adjacent_MK_function(m, k)
    mk_func.generate()

start1 = timer()
best_fit, sol = dp.bruteforce_MK_solve(mk_func)
end1 = timer()
print("Bruteforce evaluation time (ms):", 1000*(end1-start1))
print("Max fitness bruteforce:", best_fit)

start2 = timer()
best_dp_fit = dp.dp_solve_MK(mk_func)
end2 = timer()
print("DP evaluation time (ms):", 1000*(end2-start2))
print("Max fitness DP:", best_dp_fit)
