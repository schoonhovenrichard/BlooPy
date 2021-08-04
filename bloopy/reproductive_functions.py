import numpy as np
import random
from operator import itemgetter
import copy
from bitarray import bitarray

from bloopy.individual import individual

r"""
NOTE: Reproductive functions must take as input a list of individuals (parents),
     and as output a list of individuals (children).
"""

def void(parents):
    return parents

def uniform_crossover(parents):
    r"""
    Perform uniform crossover betwwen consecutive parents.
        NOTE: Automatically takes care of boundary lists.

    Args:
        parents (list(individual)): list of parent solutions.

    Returns:
        children (list(individual)): list of offspring solutions.
    """
    children = []
    random.shuffle(parents)
    if len(parents) % 2 != 0:
        raise Exception("Not implemented for uneven number of parents!")
    for k in range(len(parents)//2):
        par1 = parents[2*k]
        par2 = parents[2*k + 1]
        childs = uniform_crossover_pair(par1, par2)
        children += childs
    return children

def onepoint_crossover(parents):
    r"""
    Perform one-point crossover between consecutive parents.
        NOTE: Automatically takes care of boundary lists.

    Args:
        parents (list(individual)): list of parent solutions.

    Returns:
        children (list(individual)): list of offspring solutions.
    """
    children = []
    random.shuffle(parents)
    if len(parents) % 2 != 0:
        raise exception("not implemented for uneven number of parents!")
    for k in range(len(parents)//2):
        par1 = parents[2*k]
        par2 = parents[2*k + 1]
        childs = onepoint_crossover_pair(par1, par2)
        children += childs
    return children

def twopoint_crossover(parents):
    r"""
    Perform twopoint crossover between consecutive parents.
        NOTE: Automatically takes care of boundary lists.

    Args:
        parents (list(individual)): list of parent solutions.

    Returns:
        children (list(individual)): list of offspring solutions.
    """
    children = []
    random.shuffle(parents)
    if len(parents) % 2 != 0:
        raise Exception("Not implemented for uneven number of parents!")
    for k in range(len(parents)//2):
        par1 = parents[2*k]
        par2 = parents[2*k + 1]
        childs = twopoint_crossover_pair(par1, par2)
        children += childs
    return children

def uniform_crossover_pair(indiv1, indiv2):
    r"""
    Perform uniform crossover between two parents.
        NOTE: Automatically takes care of boundary lists.

    Args:
        indiv1 (individual): input parent solution 1.
        indiv2 (individual): input parent solution 2.

    Returns:
        children (list(individual)): list of offspring solutions.
    """
    child1 = individual(indiv1.size, bitstring=bitarray(indiv1.size*"0"), boundary_list=indiv1.boundary_list)
    child2 = individual(indiv2.size, bitstring=bitarray(indiv2.size*"0"), boundary_list=indiv2.boundary_list)
    if child1.boundary_list is not None:
        indices1 = [i for i, x in enumerate(list(indiv1.bitstring)) if x]
        indices2 = [i for i, x in enumerate(list(indiv2.bitstring)) if x]
        for k in range(len(indices1)):
            coin = random.random()
            if coin < 0.5:
                pos1 = indices1[k]
                pos2 = indices2[k]
            else:
                pos1 = indices2[k]
                pos2 = indices1[k]
            child1.bitstring[pos1] = 1
            child2.bitstring[pos2] = 1
    else:
        for k in range(indiv1.size):
            coin = random.random()
            if coin < 0.5:
                child1.bitstring[k] = indiv2.bitstring[k]
                child2.bitstring[k] = indiv1.bitstring[k]
    child1.fitness = None
    child2.fitness = None
    return [child1, child2]

def onepoint_crossover_pair(indiv1, indiv2):
    r"""
    Perform one-point crossover between two parents.
        NOTE: Automatically takes care of boundary lists.

    Args:
        indiv1 (individual): input parent solution 1.
        indiv2 (individual): input parent solution 2.

    Returns:
        children (list(individual)): list of offspring solutions.
    """
    if indiv1.boundary_list is None:
        child1 = individual(indiv1.size, bitstring=copy.deepcopy(indiv1.bitstring), boundary_list=indiv1.boundary_list)
        child2 = individual(indiv2.size, bitstring=copy.deepcopy(indiv2.bitstring), boundary_list=indiv2.boundary_list)
        a = random.randint(1, indiv1.size-1)
        for k in range(a, indiv1.size):
            child1.bitstring[k] = indiv2.bitstring[k]
            child2.bitstring[k] = indiv1.bitstring[k]
    else:
        child1 = individual(indiv1.size, bitstring=bitarray(indiv1.size*"0"), boundary_list=indiv1.boundary_list)
        child2 = individual(indiv2.size, bitstring=bitarray(indiv2.size*"0"), boundary_list=indiv2.boundary_list)
        indices1 = [i for i, x in enumerate(list(indiv1.bitstring)) if x]
        indices2 = [i for i, x in enumerate(list(indiv2.bitstring)) if x]
        crosspoint = random.randint(1, len(indices1)-1)
        for k in range(0, crosspoint):
            pos1 = indices1[k]
            pos2 = indices2[k]
            child1.bitstring[pos1] = 1
            child2.bitstring[pos2] = 1
        for k in range(crosspoint, len(indices1)):
            pos1 = indices2[k]
            pos2 = indices1[k]
            child1.bitstring[pos1] = 1
            child2.bitstring[pos2] = 1
    child1.fitness = None
    child2.fitness = None
    return [child1, child2]

def twopoint_crossover_pair(indiv1, indiv2):
    r"""
    Perform two-point crossover between two parents.
        NOTE: Automatically takes care of boundary lists.

    Args:
        indiv1 (individual): input parent solution 1.
        indiv2 (individual): input parent solution 2.

    Returns:
        children (list(individual)): list of offspring solutions.
    """
    if indiv1.boundary_list is None:
        child1 = individual(indiv1.size, bitstring=copy.deepcopy(indiv1.bitstring), boundary_list=indiv1.boundary_list)
        child2 = individual(indiv2.size, bitstring=copy.deepcopy(indiv2.bitstring), boundary_list=indiv2.boundary_list)
        a = random.randint(1, indiv1.size-2)
        b = random.randint(a, indiv1.size-1)
        for k in range(a, b):
            child1.bitstring[k] = indiv2.bitstring[k]
            child2.bitstring[k] = indiv1.bitstring[k]
    else:
        child1 = individual(indiv1.size, bitstring=bitarray(indiv1.size*"0"), boundary_list=indiv1.boundary_list)
        child2 = individual(indiv2.size, bitstring=bitarray(indiv2.size*"0"), boundary_list=indiv2.boundary_list)
        indices1 = [i for i, x in enumerate(list(indiv1.bitstring)) if x]
        indices2 = [i for i, x in enumerate(list(indiv2.bitstring)) if x]
        crosspoint1 = random.randint(1, len(indices1)-2)
        crosspoint2 = random.randint(crosspoint1, len(indices1)-1)
        for k in range(0, crosspoint1):
            pos1 = indices1[k]
            pos2 = indices2[k]
            child1.bitstring[pos1] = 1
            child2.bitstring[pos2] = 1
        for k in range(crosspoint1, crosspoint2):
            pos1 = indices2[k]
            pos2 = indices1[k]
            child1.bitstring[pos1] = 1
            child2.bitstring[pos2] = 1
        for k in range(crosspoint2, len(indices1)):
            pos1 = indices1[k]
            pos2 = indices2[k]
            child1.bitstring[pos1] = 1
            child2.bitstring[pos2] = 1
    child1.fitness = None
    child2.fitness = None
    return [child1, child2]

def mask_crossover_pair(indiv1, indiv2, mask):
    r"""
    Perform mask crossover between two parents: Cross the bits in the mask only

    Args:
        indiv1 (individual): input parent solution 1.
        indiv2 (individual): input parent solution 2.
        mask (list(int)): List of bitstring positions to cross

    Returns:
        children (list(individual)): list of offspring solutions.
    """
    child1 = individual(indiv1.size, bitstring=copy.deepcopy(indiv1.bitstring), boundary_list=indiv1.boundary_list)
    child2 = individual(indiv2.size, bitstring=copy.deepcopy(indiv2.bitstring), boundary_list=indiv2.boundary_list)
    if indiv1.boundary_list is None:
        for k in range(len(mask)):
            pos = mask[k]
            child1.bitstring[pos] = indiv2.bitstring[pos]
            child2.bitstring[pos] = indiv1.bitstring[pos]
    else:
        # Implement it by crossing within region
        for k in range(len(mask)):
            region = None
            for r in range(len(child1.boundary_list)):
                segment = child1.boundary_list[r]
                if mask[k] >= segment[0] and mask[k] <= segment[1]:
                    region = segment
                    break
            child1.bitstring[region[0]:region[1]+1] = indiv2.bitstring[region[0]:region[1]+1]
            child2.bitstring[region[0]:region[1]+1] = indiv1.bitstring[region[0]:region[1]+1]
    child1.fitness = None
    child2.fitness = None
    return [child1, child2]

def greedy_mask_crossover_pair(par, donor, mask):
    r"""
    Perform greedy mask crossover between two parents: Child is a copy of
        parent with donor bits at mask positions.

    Args:
        indiv1 (individual): input parent solution 1.
        indiv2 (individual): input parent solution 2.
        mask (list(int)): List of bitstring positions to cross.

    Returns:
        children (list(individual)): list of offspring solutions.
    """
    child = individual(par.size, bitstring=copy.deepcopy(par.bitstring), boundary_list=par.boundary_list)
    if par.boundary_list is None:
        for k in range(len(mask)):
            pos = mask[k]
            child.bitstring[pos] = donor.bitstring[pos]
    else:
        # Implement it by crossing within region
        for k in range(len(mask)):
            region = None
            for r in range(len(child.boundary_list)):
                segment = child.boundary_list[r]
                if mask[k] >= segment[0] and mask[k] <= segment[1]:
                    region = segment
                    break
            child.bitstring[region[0]:region[1]+1] = donor.bitstring[region[0]:region[1]+1]
    child.fitness = None
    return child
