import numpy as np
import random
import copy
from operator import itemgetter
import multiprocessing as mp
from bitarray import bitarray

r"""
Fitness functions must take a bitstring (bitarray()) as input,
 and return the fitness (float)
"""

def create_bitstring(val, length):
    r"""
    Helper functions for MK functions

    Args:
        val (int): integer to be encoded as bitstring.
        length (int): specify length of bitstring.
    """
    if length < 0 or val < 0:
        raise Exception("Invalid input for create bitstring")
    if 2**length -1 < val:
        raise Exception("Bitstring too short to encode value")
    bitstring = bitarray(length)
    bitstring.setall(False)
    bs = bin(val)
    for k in range(length):
        if bs[-k-1] == 'b':
            break
        bitstring[k] = bool(int(bs[-k-1]))
    return bitstring

class MK_function:
    def __init__(self, m, k, bit_mask=None, fitness_map=None):
        r"""
        Base class for MK functions.

        Args:
            m (int): Bitstring length.
            k (int): Length of subfunctions present in MK function (m must be divisible)
                        by k-1.
        """
        if m < 2 or k < 2:
            raise Exception("Invalid values for MK-function")
        self.m = m
        self.k = k
        if bit_mask is None:
            self.bit_mask = []
        else:
            self.bit_mask = bit_mask
        if fitness_map is None:
            self.fitness_map = []
        else:
            self.fitness_map = fitness_map

    def generate():
        pass

    def generate_subfunction():
        pass

    def apply_fitness():
        pass

class adjacent_MK_function(MK_function):
    def __init__(self, m, k):
        r"""
        Class for adjacent MK functions, i.e. the subfunctions are located adjactenly in
            the bitstring.

        Args:
            m (int): Bitstring length.
            k (int): Length of subfunctions present in MK function (m must be divisible)
                        by k-1.
        """
        super().__init__(m, k)

    def generate(self):
        r"""
        Generates a adjacent MK function.
        """
        for j in range(self.m):
            mk_func = self.generate_subfunction()
            self.fitness_map.append(mk_func)
            mask = []
            for i in range(self.k):
                mask.append((j + i) % self.m)
            self.bit_mask.append(mask)

    def generate_subfunction(self):
        r"""
        Helper function to generate a random subfunction.
        """
        mk_func = dict()
        size = 2**self.k
        v = list(range(1,size+1))
        random.shuffle(v)
        for i in range(size):
            bitset = create_bitstring(i, self.k)
            mk_func[tuple(bitset)] = v[i]
        return mk_func

    def get_fitness(self, input_bs):
        r"""
        Returns fitness value of input bitstring

        Args:
            input_vs (bitarray()): Bitstring to determine fitness for.

        Returns:
            fit (float): Fitness value of input bitstring
        """
        if len(input_bs) > self.m:
            raise Exception("Input string should be of size {}".format(self.m))
        fit = 0.0
        for j in range(len(self.fitness_map)):
            mk_func = self.fitness_map[j]
            current_mask = self.bit_mask[j]
            getter = itemgetter(*current_mask)
            bitset = getter(input_bs)
            fit += mk_func[tuple(bitset)]
        return fit

class random_MK_function(adjacent_MK_function):
    def __init__(self, m, k):
        r"""
        Class for randomized MK functions, i.e. the input bits for each subfunction are 
            located at random positions throughout the bitstring.

        Args:
            m (int): Bitstring length.
            k (int): Length of subfunctions present in MK function (m must be divisible)
                        by k-1.
        """
        super().__init__(m, k)
        self.adj_bit_mask = None

    def set_random_bitmask(self):
        r"""
        Set the random positions for inputs for each subfunction.
        """
        if self.adj_bit_mask is None:
            self.adj_bit_mask = copy.deepcopy(self.bit_mask) # Save original adjacent mask

        permutation = list(range(self.m))
        random.shuffle(permutation)
        for i in range(len(self.bit_mask)):
            for j in range(len(self.bit_mask[i])):
                pos = self.bit_mask[i][j]
                self.bit_mask[i][j] = permutation[pos] 

    def generate(self):
        r"""
        Generates a randomized MK function.
        """
        for j in range(self.m):
            mk_func = self.generate_subfunction()
            self.fitness_map.append(mk_func)
            mask = []
            for i in range(self.k):
                mask.append((j + i) % self.m)
            self.bit_mask.append(mask)
        self.set_random_bitmask()
