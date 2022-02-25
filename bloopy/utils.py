import math
import numpy as np
import random
from bitarray import bitarray
import os

from bloopy.individual import individual

def create_bitstring_from_int(val, length):
    r"""
    Create a bitarray initalized from an integer value.

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

class discrete_space:
    r""" Class that stores the optimization space.
    Contains dictionary of possible values for each variable,
     and function that returns fitness for each possible combination.

    Args:
        fitness_function (list[] -> float): Function that scores
                fitness of bitstrings.
        variables (dict): Dictionary with for every variable name
                a list of possible values.
        missing_fit (float): (optional) Fitness value for combinations
                that are missing in the fitness functions. Default is
                None, which throws an error if an unknown combination
                is evaluated.
    """
    def __init__(self, fitness_func, variables):
        self.variables = variables

        if isinstance(fitness_func, dict):
            # If it is a dictionary, turn it into a function
            self.fdict = fitness_func
            self.fitness_func = lambda x: self.fdict[x]
        elif hasattr(fitness_func, '__call__'):
            # If it is already a function, do nothing
            self.fitness_func = fitness_func
        else:
            raise Exception("fitness_func not of type function or dictionary")

    def get_bitstring_as_paramvec(self, bitstring):
        ## Convert bitstring to vector of variable values
        vector = []
        start_ind = 0
        bls = list(bitstring)
        for vals in self.variables.values():
            end_ind = start_ind + len(vals)
            indices = [i for i, x in enumerate(bls[start_ind:end_ind]) if x]
            # Check that the bitstring is valid, i.e. has only one parameter value chosen
            if len(indices) != 1:
                raise Exception("Asked to compute fitness of invalid bitstring!")
            vector.append(vals[indices[0]])
            start_ind = end_ind
        return vector

    def fitness(self, bitstring):
        vector = self.get_bitstring_as_paramvec(bitstring)
        return self.fitness_func(vector)

    def isvalid(self, bitstring):
        r""" Safeguards against invalid bitstrings
        """
        start_ind = 0
        valid = True
        for key in self.variables.keys():
            end_ind = start_ind + len(self.variables[key])
            val = sum(list(bitstring[start_ind:end_ind]))
            if val != 1:
                print("Invalid for", key, start_ind, end_ind)
                valid = False
            start_ind = end_ind
        return valid

class bitstring_as_discrete:
    def __init__(self, searchspace, bit_fit_func):
        self.sspace = searchspace
        self.bit_fit_func = bit_fit_func

    def get_fitness(self, x):
        # As it is orignally a bitstring problem, we know each
        #  variable only takes 2 values: 0 or 1.
        if len(x) % 2 != 0:
            raise Exception("Something wrong, should be multiple of 2")
        bitstring = bitarray(len(x)//2)
        bitstring.setall(False)
        for i in range(len(bitstring)):
            if sum(list(x[2*i:2*i+2])) != 1:
                raise Exception("Error when encoding solution")
            if x[1+2*i]:
                bitstring[i] = True
        return self.bit_fit_func(bitstring)

def create_bitstring_searchspace(bs_size):
    searchspace = dict()
    for i in range(bs_size):
        searchspace[str(i)] = [0,1]
    return searchspace

def generate_boundary_list(sspace):
    r"""
    Generate the segments for encoding a search space
    as a bitstring.
    """
    boundary_list = []
    stidx = 0
    for vls in list(sspace.values()):
        boundary_list.append((stidx, stidx + len(vls) - 1))
        stidx += len(vls)
    return boundary_list

def create_cleaned_parameter_space(cached_sols, reduced_sspace, orig_sspace):
    nr_orig_params = len(orig_sspace.keys())
    nr_tunable_params = len(reduced_sspace.keys())
    cached_solutions = []

    indices_to_tune = []
    it = 0
    for key in list(orig_sspace.keys()):
        if key in list(reduced_sspace.keys()):
            indices_to_tune.append(it)
        it += 1

    for key in cached_sols.keys():
        settings = [int(x) for x in key.split(",")]
        if len(settings) == nr_tunable_params:
            cached_solutions.append(settings)
        elif len(settings) == nr_orig_params:
            vect = []
            for j in indices_to_tune:
                vect.append(settings[j])
            cached_solutions.append(vect)
        else:
            raise Exception("Something wrong with cached keys and searchspace")
    return cached_solutions

def convert_paramspace_to_bitstringspace(paramspace, sspace):
    boundary_list = generate_boundary_list(sspace)
    vals = list(sspace.values())

    bs_size = 0
    for vls in vals:
        bs_size += len(vls)

    restricted_space = []
    for var in paramspace:
        bs = bitarray(bs_size*"0")
        for i in range(len(var)):
            for j in range(len(vals[i])):
                if var[i] == vals[i][j]:
                    bs[boundary_list[i][0]+j] = 1
        restricted_space.append(bs.to01())
    return restricted_space

def clean_up_searchspace(sspace):
    r"""
    Remove unnecessary keys from the search space in-place.
    """
    recurse = False
    newspace = sspace.copy()
    for key in newspace.keys():
        possible_values = len(newspace[key])
        if possible_values <= 1:
            #print("Removing {} from searchspace...".format(key))
            del newspace[key]
            recurse = True
            break
    if recurse:
        rspace = clean_up_searchspace(newspace)
    else:
        rspace = newspace
    return rspace

def calculate_bitstring_length(sspace):
    bsize = 0
    for key in sspace.keys():
        if len(sspace[key]) <= 1:
            raise Exception("Search space not properly pre-processed")
        bsize += len(sspace[key])
    return bsize

def set_bitstring(sol, arr):
    if sol.boundary_list is None:
        raise Exception("Method not implemented for standard bitstrings")
    if len(arr) != len(sol.boundary_list):
        raise Exception("Invalid settings for bitstring given!")
    indices = [i for i, x in enumerate(list(sol.bitstring)) if x]
    for k in range(len(arr)):
        sol.bitstring[indices[k]] = 0 # Set old one to 0
        sol.bitstring[arr[k]] = 1 # set new one to 1

def generate_population(population_size, ffunc, sspace):
    splits = generate_boundary_list(sspace)
    input_pop = []
    count = 0
    bsize = 0
    for vals in list(sspace.values()):
        bsize += len(vals)
    for i in range(population_size):
        test_indiv = individual(bsize, boundary_list=splits)
        test_indiv.fitness = ffunc(test_indiv.bitstring)
        input_pop.append(test_indiv)
        if test_indiv.fitness < 100000000:
            count +=1
    #print("\nGenerated {0} individuals with {1} sensible settings".format(population_size, count))
    return input_pop

def generate_log_population(population_size, ffunc, sspace):
    logbsize = 0
    for var in sspace:
        bslen = len(sspace[var])
        loglen = int(math.ceil(np.log2(bslen)))
        logbsize += loglen
    input_pop = []
    count = 0
    for i in range(population_size):
        fullarr = []
        for var in sspace:
            bslen = len(sspace[var])
            loglen = int(math.ceil(np.log2(bslen)))
            rand = random.randint(0, bslen-1)
            arrlog = np.zeros(shape=(loglen,), dtype=np.bool)
            val = rand
            while val > 0:
                k = int(np.log2(val))
                arrlog[k] = True
                val -= 2**k
            for x in arrlog.tolist():
                fullarr.append(x)
        bslog = bitarray(fullarr)
        test_indiv = individual(logbsize, bitstring=bslog, fitness=None)
        input_pop.append(test_indiv)
    return input_pop

class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    NOTE: Necessary to surpress Fortran warnings for scipy.optimize COBYLA.
    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds =  [os.open(os.devnull,os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        #os.dup2(self.null_fds[0],1)
        os.dup2(self.null_fds[1],2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0],1)
        os.dup2(self.save_fds[1],2)
        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)
