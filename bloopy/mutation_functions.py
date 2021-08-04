import numpy as np
import random

r"""
NOTE: Mutation functions must take a list of individual objects as input,
     and mutate the individuals in place, i.e. return None.
"""

def trivial_mutate(indivs):
    r"""
    Perform no mutation

    Args:
        None
    """
    return None

def point_mutate(indivs):
    r"""
    Perform a single bit flip at a random position.

    Args:
        indivs (list(individuals)): Population of individuals on
                which to perform mutation
    """
    for indiv in indivs:
        splits = indiv.boundary_list
        if splits is None:
            pos = random.randint(0, len(indiv.bitstring)-1)
            indiv.bitstring[pos] = not indiv.bitstring[pos]
        else:
            indices = [i for i, x in enumerate(list(indiv.bitstring)) if x]
            substr = random.randint(0, len(splits)-1)
            indiv.bitstring[indices[substr]] = 0
            pos = random.randint(splits[substr][0], splits[substr][1]) 
            indiv.bitstring[pos] = 1
        indiv.fitness = None
