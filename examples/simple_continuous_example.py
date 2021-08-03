import numpy as np
import itertools as it

from simple_discrete_example import categorical_fitness
import algorithms.dual_annealing as dsa
import utils

### Construct some categorical discrete space
searchspace = {"x1": [1,2,3,4,5,6],
               "x2": ["foo", "bar"],
               "x3": [16, 32, 64, 128],
               "x4": ["a", "b", "c", "d", "e"]}

# Continuous algorithms require a search space to operate
categorical_fit = categorical_fitness(searchspace)
disc_space = utils.discrete_space(categorical_fit.fitness, searchspace)


## Run dual annealing
# supported_methods = ['COBYLA','L-BFGS-B','SLSQP','CG','Powell','Nelder-Mead', 'BFGS', 'trust-constr']
method = "trust-constr"
iterations = 10000
minmax = -1 # -1 for minimization problem, +1 for maximization problem
if minmax == 1:
    optfit = len(categorical_fit.possible_xs)
elif minmax == -1:
    optfit = 1
maxfeval = 100000 # Number of unique fitness queries MLS is allowed

test_dsa = dsa.dual_annealing(disc_space.fitness,
        minmax,
        searchspace,
        method=method)

best_fit, _, fevals = test_dsa.solve(max_iter=iterations,
            max_time=10,#seconds
            stopping_fitness=optfit,
            max_funcevals=maxfeval)
print("Best fitness found:", best_fit, "in", fevals, "evaluations | optimal fitness:", optfit)
