import random
from timeit import default_timer as timer
from bitarray import bitarray
import numpy as np

class individual:
    def __init__(self, size, bitstring=None, fitness=None, boundary_list=None):
        r"""
        Individual that represents a solution in the evolutionary algorithm.

        Args:
            size (int): length of the bitstring that encodes a solution
            bitstring (bitarray()): solution instance encoded as bitstring
            fitness (float): fitness of the encoded solution
            boundary_list (list[tuple(int)]): In the case of the bitstring representing
                a chosen element from a certain collection of lists, only one 1
                is allowed in each region. The region has length equal to the length
                of possible options
                    E.g.: Suppose possible choices are [[16,32,64],['foo','bar']],
                    the bitstring has length 5, and only one of position [0,1,2], 
                    can be a 1, and only one of [3,4]. So the boundary list is 
                    [(0,2),(3,4)] to markt he start and end of each region.
        """
        if size < 1 or not isinstance(size, int):
            raise Exception("size must be a positive integer")
        self.size = size
        self.fitness = fitness
        self.boundary_list = boundary_list
        self.bitstring = bitstring
        if self.bitstring is not None:
            if not isinstance(bitstring, bitarray):
                raise Exception("bitstring must be of type bitarray")
            if len(self.bitstring) != self.size:
                raise Exception("bitstring length and size parameter do not match")
        else:
            self.generate_random()

    def generate_random(self, boundary_list=None):
        r"""
        Generates a randomly initialized bitstring and resets the fitness.
        """
        if self.boundary_list is None:
            tup = np.random.choice(a=[False, True], size=(self.size,), p=[0.5, 0.5])
            self.bitstring = bitarray(tup.tolist())
            self.fitness = None
        else:
            arr = np.zeros(shape=(self.size,), dtype=np.bool)
            for lb, ub in self.boundary_list:
                pos = random.randint(lb, ub)
                arr[pos] = True
            self.bitstring = bitarray(arr.tolist())
            self.fitness = None


class continuous_individual:
    def __init__(self, float_solution, searchspace, fitness=None, scaling=None):
        r"""
        Individual that represents a solution in the evolutionary algorithm
         that can be applied in real-valued continuous algorithms.

        Args:
            float_solution (list(float)): list that contains the multivariable
                real-valued solution.
            searchspace (dict(params, vals)): Dict that contains the
                tunable variables and their possible values.
            fitness (float): fitness of the encoded solution

        Attirbutes:
            size (int): Defines length of bitstring.
            boundary_list (list[tuple(int)]): Defines segments in bitstring.
            bitstring (bitarray()): solution instance encoded as bitstring
            nearest_solution (list(val)): Closest valid/existing solution
                to the real-valued variable in the searchspace.

        Used in algorithms:
            - Local minimization
            - Basin Hopping
            - PSO
            - Differential evolution
            - Dual annealing
            - CMA-ES

        ### NOTE: Snapping of values in continuous domain is copied from
                kernel tuner code.
        """
        self.float_solution = float_solution
        self.sspace = searchspace
        self.fitness = fitness
        self.scaling = scaling

        self.size = None
        self.boundary_list = None
        self.bitstring = None
        self.nearest_solution = None

        self.generate_boundary_list()
        self.set_bitstring()

    def calculate_size(self):
        correct_size = 0
        for vls in list(self.sspace.values()):
            correct_size += len(vls)
        self.size = correct_size

    def generate_boundary_list(self):
        boundary_list = []
        stidx = 0
        for vls in list(self.sspace.values()):
            boundary_list.append((stidx, stidx + len(vls) - 1))
            stidx += len(vls)
        if self.boundary_list is not None:
            raise Exception("Boundary list already set, should not be overwritten")
        self.boundary_list = boundary_list

    def set_nearest_solution(self):
        r"""Helper function that for each parameter selects the 
            closest valid value in the search space.
        """
        params = []
        for i, k in enumerate(self.sspace.keys()):
            values = np.array(self.sspace[k])
            idx = np.abs(values - self.float_solution[i]).argmin()
            params.append(int(values[idx]))
        self.nearest_solution = params

    def set_scaled_nearest_solution(self):
        r"""Helper function that for each parameter selects the 
            closest valid value in the search space, assuming 
            scaled real-valued input vectors.
        """
        x = self.float_solution
        x_u = [i for i in x]
        values = self.sspace.values()
        for i, v in enumerate(values):
            pad = 0.5*self.scaling     # use when interval is [0, eps*len(v)]
            linspace = np.linspace(pad, (self.scaling*len(v))-pad, len(v))
            idx = np.abs(linspace-x[i]).argmin()
            idx = min(max(idx, 0), len(v)-1)
            x_u[i] = v[idx]
        self.nearest_solution = x_u

    def set_bitstring(self):
        r"""Converts the real-valued vector to a bitstring encoding.
        """
        self.calculate_size()
        if self.scaling is not None:
            self.set_scaled_nearest_solution()
        else:
            self.set_nearest_solution()
        bs = bitarray(self.size*"0")
        vals = list(self.sspace.values())
        for i in range(len(self.nearest_solution)):
            for j in range(len(vals[i])):
                if self.nearest_solution[i] == vals[i][j]:
                    bs[self.boundary_list[i][0]+j] = 1
        self.bitstring =  bs
