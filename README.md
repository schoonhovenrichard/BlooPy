![Alt text](imgs/bloopy_logo.png?raw=true "Title")

# BlooPy: Python library for black-box discrete optimization

## Installation
Create conda environment:
``` shell
conda create -n bloopy
conda activate bloopy 
```

### Dependencies
**BlooPy** requires the following packages to function:
- bitarray

which can be installed by:
``` shell
conda install -c anaconda bitarray 
```

Finally, install **BlooPy** by running
``` shell
pip install -e . 
```

## Features
### Implemented algorithms
- Genetic Algorithm
- Genetic Local Search
- Multi-start Local Search (4 different versions)
- Iterative Local Search (4 different versions)
- Tabu Search (2 different versions)
- Simulated Annealing
- Dual Annealing
- Particle Swarm Optimization
- Basin Hopping
- Differential Evolution (and discrete version in beta test)
- Continuous local minimization algorithms
- Linkage Tree Genetic Algorithm (and greedy version) [[1]](#1).

**BlooPy** is a fast discrete optimization package because it translates the optimization to a bitstring encoding, and uses efficient bitstring procedures implemented in C. 

In addition to fitness functions of type "bitarray() --> float", **BlooPy** is also usable with fitness functions on discrete vectors "list[] --> float" or fitness dictionaries. The ```utils.discrete_space``` performs encoding automatically performs the translation from fitness dict or vector-based function to bitstring encoding for the use. See examples/example_discrete_space.py.

**BlooPy** allows users to assemble algorithms by passing components such as mutation functions, reproduction functions to the algorithm at initialization: 

```python
fitness_functions # Input: bitstring of bitarray() type, Output: fitness (float)
mutation_functions # Input: list of individuals, Output: None
reproductive functions # Input: list of individuals (parents), Output: list of individuals (children)
selection_functions # Input: {list(individuals), list(individuals), float}. The float is -1 or 1 depending on whether we are minimizing or maximizing
```

Additional components can easily be added by the user as long as the function has the same signature as specified.

## Usage

To learn more about the functionality of the package check out our
examples folder. As a test suite, we have included code to generate adjacent, and randomized MK functions [[2]](#2). The examples folder contains scripts to test each algorithm on randomized MK functions. 

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

### Back-end encoding solutions
Here is some background information on how **BlooPy** operates. The algorithms are intended for discrete optimization problems, and they work on bitstrings. **BlooPy** implements two types of bitstring. 

- Normal bitstrings which can take on any permutation. In this case, **BlooPy** creates ```individual(..., boundary_list=None)``` objects.
- Bounded bitstrings where only a single 1 can be present in each segment. The segments are defined by supplying a list of start- and endpoints of the segments: ```individual(..., boundary_list=[(0,4),(5,7),(8,12),..])```.
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

All the optimization algorithms will handle boundary lists properly as keyword arguments. If a boundary list is provided, the algorithms will only consider suitable candidates.

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
