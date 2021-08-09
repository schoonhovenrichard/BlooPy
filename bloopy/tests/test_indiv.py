import pytest
from hypothesis import given, settings, assume
import hypothesis.strategies as st
import random
import numpy as np

import bloopy.individual as indiv

@given(st.integers(min_value=1, max_value=1e6))
@settings()
def test_individual_hypo_check_size(size):
    temp = indiv.individual(size)
    siz = len(temp.bitstring)
    assert siz == size, "Failed test individual: incorrect bitstring size"

@given(st.integers(min_value=1, max_value=1e2))
@settings(deadline=1000, max_examples=10)
def test_individual_hypo_generate_random_no_bound(size):
    # This is difficult to check, we look at the statistics of each bit
    eps = 0.1
    stats = np.zeros(shape=(size,),dtype=float)
    runs = size*100
    for t in range(runs):
        temp = indiv.individual(size)
        bstr = np.array(temp.bitstring.tolist())
        stats += bstr
    stats = stats/float(runs)
    assert max(abs(stats-0.5)) < eps, "Failed test individual: generate random no bounds failed statistics"

def test_individual_check_size(tries=10):
    for t in range(tries):
        size = random.randint(1,10000)
        temp = indiv.individual(size)
        siz = len(temp.bitstring)
        assert siz == size, "Failed test individual: incorrect bitstring size"

def test_individual_generate_random_no_bound(tries=10, eps=0.1):
    # This is difficult to check, we look at the statistics of each bit
    for t in range(tries):
        size = random.randint(8, 32)
        stats = np.zeros(shape=(size,),dtype=float)
        runs = size * 100
        #stdev = np.sqrt(0.25/float(runs))
        #print('stdev:', stdev)
        for t in range(runs):
            temp = indiv.individual(size)
            bstr = np.array(temp.bitstring.tolist())
            stats += bstr
        stats = stats/float(runs)
        #print("max:", max(abs(stats-0.5)),'\n')
        #print(sum(abs(stats-0.5) < 2*stdev)/float(len(stats)),'\n')
        assert max(abs(stats-0.5)) < eps, "Failed test individual: generate_random_no_bounds failed statistics"

def test_individual_generate_random_with_bound(tries=10, eps=0.1):
    # This is difficult to check, we look at the statistics of each bit
    for t in range(tries):
        size = random.randint(6, 16)
        nrparts = size//3
        parts = random.sample(range(size), nrparts)
        parts.sort()
        bounds = [(0, parts[1]-1)] + [(parts[i], parts[i+1]-1) for i in range(1,nrparts-1)] + [(parts[-1], size-1)]

        stats = np.zeros(shape=(size,),dtype=float)
        runs = size * 100
        for t in range(runs):
            temp = indiv.individual(size, boundary_list=bounds)
            bstr = np.array(temp.bitstring.tolist())
            stats += bstr
        stats = stats/float(runs)
        for lb, ub in bounds:
            mean = stats[lb:ub+1].mean()
            assert max(abs(stats[lb:ub+1]-mean)) < eps, "Failed test individual: generate_random_with_bounds failed statistics"

def test_cont_individual_size(tries=10):
    for t in range(tries):
        # Generate random search space
        nr_vars = random.randint(1,100)
        searchspace = dict()
        size = 0
        for i in range(nr_vars):
            nr_vals = random.randint(1, 16)
            vals = random.sample(range(2**nr_vals), nr_vals)
            vals.sort()
            searchspace["x{}".format(i)] = vals
            size += nr_vals

        # Generate boundary list
        boundary_list = []
        vector = []
        stidx = 0
        for vls in list(searchspace.values()):
            boundary_list.append((stidx, stidx + len(vls) - 1))
            stidx += len(vls)
            # Generate random float_solution
            randval = float(vls[0] + random.random()*vls[-1])
            vector.append(randval)

        temp = indiv.continuous_individual(vector, searchspace)
        siz = len(temp.bitstring)
        assert siz == size, "Failed test individual: incorrect bitstring size"

#TODO: Check size function in continuous individual
#TODO: Check float solution function in continuous individual
#TODO: Fix all the docstrings

if __name__ == '__main__':
    test_individual_generate_random_with_bound()
    test_individual_generate_random_no_bound()
    test_cont_individual_size()
    test_individual_check_size()
