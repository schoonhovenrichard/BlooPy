<p align="center">
  <br>
  <a href="https://github.com/schoonhovenrichard/BlooPy"><img src="./imgs/bloopy_logo.png" height="180"></a>
  <br>
</p>


<br>

---

<h2 align="center">
  BlooPy: Black-box optimization Python for bitstring, categorical, and numerical discrete problems with local, and population-based algorithms.
</h2>

<br>

## Installation
Install **BlooPy** by running
``` shell
git clone https://github.com/schoonhovenrichard/BlooPy.git
cd BlooPy/
pip install -e . 
```

### Dependencies
**BlooPy** requires the following packages to function:
- bitarray
- pyswarms

## Implemented algorithms
### Discrete local search algorithms
- Multi-start Local Search (4 different versions)
- Iterative Local Search (4 different versions)
- Tabu Search (2 different versions)
- Simulated Annealing

### Discrete population-based algorithms
- Genetic Algorithm
- Genetic Local Search
- Linkage Tree Genetic Algorithm (and greedy version) [[1]](#1).

### Continuous (real-valued) optimization algorithms
- Dual Annealing
- Particle Swarm Optimization
- Basin Hopping
- Differential Evolution (and discrete version in beta test)
- Continuous local minimization algorithms

**BlooPy** is a fast discrete optimization package because it translates the optimization to a bitstring encoding, and uses efficient bitstring procedures implemented in C. 

In addition to fitness functions of type "bitarray() --> float", **BlooPy** is also usable with fitness functions on discrete vectors "list[] --> float" or fitness dictionaries. The ```utils.discrete_space``` translates fitness dict or vector-based function to bitstring encoding for the use. See examples/example_discrete_space.py.

**BlooPy** allows users to assemble components for population-based algorithms by passing components such as mutation functions, reproduction functions to the algorithm at initialization: 

```python
fitness_functions # Input: bitstring of bitarray() type, Output: fitness (float)
mutation_functions # Input: list of individuals, Output: None
reproductive functions # Input: list of individuals (parents), Output: list of individuals (children)
selection_functions # Input: {list(individuals), list(individuals), float}. The float is -1 or 1 depending on whether we are minimizing or maximizing
```

Additional components can easily be added by the user as long as the function has the same signature as specified.

## Examples

To learn more about the functionality of the package check out our
examples folder. As a test suite, we have included code to generate adjacent, and randomized MK functions [[2]](#2). The examples folder contains scripts to test each algorithm on randomized MK functions. 

<details open>
<summary><b>Simple local search on bitstring problem</b></summary>

```python
import numpy as np
from bitarray.util import ba2int
import algorithms.local_search as mls

# Define fitness function and searchspace
N = 20 # size of bitstring

fitness_values = np.arange(1, 1 + 2**N)
np.random.shuffle(fitness_values)
fitness_func = lambda bs: fitness_values[ba2int(bs)]
print("Size of search space:", len(fitness_values))

# Configure Local Search algorithm (RandomGreedy MLS in this case)
iterations = 10000 # Max number of random restarts
minmax = -1 # -1 for minimization problem, +1 for maximization problem
maxfeval = 100000 # Number of unique fitness queries MLS is allowed

test_mls = mls.RandomGreedyMLS(fitness_func,
        N,
        minmax)

best_fit, _, fevals = test_mls.solve(iterations,
            max_time=10,#seconds
            stopping_fitness=1,#1 is optimal value so we can stop
            max_funcevals=maxfeval,
            verbose=True)
print("Best fitness found:", best_fit, "in", fevals, "evaluations | optimal fitness:", 1)
```
</details>

<details>
<summary><b>GA on bitstring randomized MK function</b></summary>

Let's run a genetic algorithm (see examples/example_ga.py). Firstly, import the modules and set the seed for reproducibility:

```python
import random
import fitness_functions as ff
import dynamic_programming as dp
import genetic_algorithm as ga
import mutation_functions as mut
import reproductive_functions as rep
import selection_functions as sel

random.seed(1234567)
```

Generate an adjacent or randomized MK function for testing. For this type of fitness function we have supplied a solver which uses dynamic programming.

```python
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
```

**BlooPy** allows users to assemble the components of an evolutionary algorithm separately, which can then be passed as functions at initialization:

```python
fitness_func = adj_mk_func.get_fitness
population_size = 500
reproductor = rep.twopoint_crossover
selector = sel.tournament2_selection
bitstring_size = m
test_ga = ga.genetic_algorithm(fitness_func,
            reproductor,
            selector,
            population_size,
            bitstring_size,
            min_max_problem=1, # This is a maximzation problem
            input_pop=None)
```

Run the GA to solve the problem and choose termination conditions:

```python
x = test_ga.solve(min_variance=0.1,
            max_iter=1000,
            no_improve=300,
            max_time=15,#seconds
            stopping_fitness=0.98*best_dp_fit,#fraction of optimum we want (optional)
            max_funcevals=200000)
print("Best fitness:",x[0],", fraction of optimal {0:.4f}".format(x[0]/float(best_dp_fit)))
```

</details>

<details>
<summary><b>Local search on categorical optimization problem</b></summary>

Let's run a GreedyMLS algorithm on an example discrete categorical optimization problem. For this, we will use the ```utils.discrete_space``` class to map the categorical vectors to bitstring encoding automatically. Firstly, lets define some categorical search space and give each possibility a random fitness.


In addition to fitness functions of type "bitarray() --> float", **BlooPy** is also usable with fitness functions on discrete vectors "list[] --> float" or fitness dictionaries. The ```utils.discrete_space``` translates fitness dict or vector-based function to bitstring encoding for the use. See examples/example_discrete_space.py.

```python
import numpy as np
from bitarray.util import ba2int
import itertools as it

import algorithms.local_search as mls
import utils

### Construct some categorical discrete space
searchspace = {"x1": [1,2,3,4,5,6],
               "x2": ["foo", "bar"],
               "x3": [16, 32, 64, 128],
               "x4": ["a", "b", "c", "d", "e"]}
ssvalues = list(searchspace.values())

### Give all possible (x1,x2,x3,x4) a random fitness value
var_names = sorted(searchspace)
possible_xs = list(it.product(*(searchspace[key] for key in var_names)))
print("Size of search space:", len(possible_xs))

# Define fitness function
fitness_values = np.arange(1, 1 + len(possible_xs))
np.random.shuffle(fitness_values)

    for j, var in enumerate(vec):
        vals = ssvalues[j]
        for k, x in enumerate(vals):
            if x == var:
                ls.append(k+it)
                bs[k+it] = True
                break

# Calculate bitstring size
boundary_list = utils.generate_boundary_list(searchspace)
bsize = utils.calculate_bitstring_length(searchspace)
print("Size of bitstring:", bsize)

def map_listvariable_to_index(vec):
    r"""For discrete categorical problems, bitstrings are implemented
      as segments where one bit is active in each segment, and this bit
      designates the parameter value for that variable."""
    # Find indices of active bits:
    indices = []
    it = 0
    for j, var in enumerate(vec):
        vals = ssvalues[j]
        for k, x in enumerate(vals):
            if x == var:
                indices.append(k+it)
                break
    multip = len(possible_xs)
    index = 0
    for i, key in enumerate(searchspace.keys()):
        add = indices[i]
        multip /= len(searchspace[key])
        add *= multip
        index += add
    return int(index)

# Map each entry to a unique index, which points to a random fitness value
fitness_func = lambda vec: fitness_values[map_listvariable_to_index(vec)]

# Create discrete space class
disc_space = utils.discrete_space(fitness_func, searchspace)
```

Next, configure the Greedy local search algorithm and solve the problem.

```python
### Configure Local Search algorithm (RandomGreedy MLS in this case)
iterations = 10000 # Max number of random restarts
minmax = -1 # -1 for minimization problem, +1 for maximization problem
if minmax == 1:
    optfit = len(possible_xs)
elif minmax == -1:
    optfit = 1
maxfeval = 100000 # Number of unique fitness queries MLS is allowed

test_mls = mls.RandomGreedyMLS(disc_space.fitness,
        bsize,
        minmax,
        searchspace=searchspace)

best_fit, _, fevals = test_mls.solve(iterations,
            max_time=10,#seconds
            stopping_fitness=optfit,#1 is optimal value so we can stop
            max_funcevals=maxfeval,
            verbose=False)
#            verbose=True)
print("Best fitness found:", best_fit, "in", fevals, "evaluations | optimal fitness:", optfit)
```
</details>

### Back-end encoding solutions
Some background information on how **BlooPy** operates: the algorithms are intended for discrete optimization problems, and they work on bitstrings. **BlooPy** implements two types of bitstring. 

- Normal bitstrings which can take on any permutation. In this case, **BlooPy** creates ```individual(..., boundary_list=None)``` objects.
- Bounded bitstrings where only a single 1 can be present in each segment. The segments are defined by supplying a list of start- and endpoints of the segments: ```individual(..., boundary_list=[(0,4),(5,7),(8,12),..])```.

The first kind of bitstring is for bitstring based optimization problems. The second is used to encode finite discrete optimization problems. The bitstring is divided into segments with length equal to the number of parameter possibilities per variable. The i-th parameter value is selected by the bit that is turned on.

- Real-valued solutions. Instead of a discrete solution, **BlooPy** supports continuous individuals which automatically take care of the conversion between the discrete optimization problem, and the continuous solver.

Bounded bitstrings can be used when the optimization tasks is to find the optimal setttings when parameters which can each be selected from a finite list:

```python
"""Suppose that possible choices of a problem are to select (x,y) 
from [16,32,64] and ['foo','bar']. In that case the bounded 
bitstring has length 5. The first segment consists of positions 
[0,1,2] and the second of [3,4].
"""
import individual as indiv

candidate = indiv.individual(5, boundary_list=[(0,2),(3,4)])
```

## Articles

- To be published...

## Authors and contributors
* **Richard Schoonhoven** - *Initial work*

## Cite
TODO, zenodo reference?

## Contribute

Contributions are always welcome! Please submit pull requests against the ```master``` branch.
Please read the [contribution guidelines](contributing.md) first.

## References
<a id="1">[1]</a> 
Thierens, Dirk (2010).
The linkage tree genetic algorithm.
International Conference on Parallel Problem Solving from Nature. Springer, Berlin, Heidelberg, 2010.

<a id="2">[2]</a> 
Wright, Alden H., Richard K. Thompson, and Jian Zhang (2000).
The computational complexity of NK fitness functions.
IEEE Transactions on Evolutionary Computation 4.4 (2000): 373-379.

## License

This project is licensed under the GNU General Public License v3 - see the [LICENSE.md](LICENSE.md) file for details.
