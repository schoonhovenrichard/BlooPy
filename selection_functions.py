import numpy as np
import random
from bitarray.util import count_xor

r"""
NOTE: Selection functions must take 3 arguments:
        - parents (list(individuals)),
        - children (list(individuals)),
        - minmax (int) <-- 1 for maximization problem, -1 for minimization
"""

def similar(c1bs, c2bs):
    r"""
    Helper function to compute similarity between to bitstrings.
    """
    similarity = c1bs.size - count_xor(c1bs.bitstring, c2bs.bitstring)
    return similarity

def RTS(parents, children, minmax, w=150):
    r"""
    Perform Restricted Tournament Selection (RTS) to obtain
    next generation from parents and children.

    Args:
        parents (list(individual)): input parent population.
        children (list(individual)): input children populations.
        minmax (int): 1 if maximization problem, -1 for minization.
        w (int): (optional) Defines search size for RTS to find
                rival solutions.

    Returns:
        new_generation (list(individual)): next generation.
    """
    new = []
    for i in range(len(parents)):
        c1 = children[i]
        rival = None
        for k in range(w):
            pos = random.randint(0, len(parents)-1)
            if rival is None:
                rival = parents[pos]
            elif similar(c1, rival) < similar(c1, parents[pos]):
                rival = parents[pos]
        if minmax*rival.fitness < minmax*c1.fitness:
            new.append(c1)
    total = parents + new
    total.sort(key=lambda x: x.fitness, reverse=(minmax>0))
    return total[:len(parents)]

def select_best_half(parents, children, minmax):
    r"""
    Select best half from parents and children.

    Args:
        parents (list(individual)): input parent population.
        children (list(individual)): input children populations.
        minmax (int): 1 if maximization problem, -1 for minization.

    Returns:
        new_generation (list(individual)): next generation.
    """
    total = parents + children
    total.sort(key=lambda x: x.fitness, reverse=(minmax>0))
    return total[:len(parents)]

def tournament2_selection(parents, children, minmax):
    r"""
    Selects next generation by performing random size-2 tournaments.

    Args:
        parents (list(individual)): input parent population.
        children (list(individual)): input children populations.
        minmax (int): 1 if maximization problem, -1 for minization.

    Returns:
        new_generation (list(individual)): next generation.
    """
    new = []
    total = parents + children
    tournament_size = 2
    for i in range(len(parents)):
        pos = random.randint(0, len(total)-1)
        champion = total[pos]
        for j in range(tournament_size):
            pos2 = random.randint(0, len(total) -1)
            challenger = total[pos2]
            if challenger.fitness*minmax > champion.fitness*minmax:
                champion = challenger
        new.append(champion)
    return new

def tournament4_selection(parents, children, minmax):
    r"""
    Selects next generation by performing random size-4 tournaments.

    Args:
        parents (list(individual)): input parent population.
        children (list(individual)): input children populations.
        minmax (int): 1 if maximization problem, -1 for minization.

    Returns:
        new_generation (list(individual)): next generation.
    """
    new = []
    total = parents + children
    tournament_size = 4
    for i in range(len(parents)):
        pos = random.randint(0, len(total)-1)
        champion = total[pos]
        for j in range(tournament_size):
            pos2 = random.randint(0, len(total) -1)
            challenger = total[pos2]
            if challenger.fitness*minmax > champion.fitness*minmax:
                champion = challenger
        new.append(champion)
    return new

def tournament8_selection(parents, children, minmax):
    r"""
    Selects next generation by performing random size-8 tournaments.

    Args:
        parents (list(individual)): input parent population.
        children (list(individual)): input children populations.
        minmax (int): 1 if maximization problem, -1 for minization.

    Returns:
        new_generation (list(individual)): next generation.
    """
    new = []
    total = parents + children
    tournament_size = 8
    for i in range(len(parents)):
        pos = random.randint(0, len(total)-1)
        champion = total[pos]
        for j in range(tournament_size):
            pos2 = random.randint(0, len(total) -1)
            challenger = total[pos2]
            if challenger.fitness*minmax > champion.fitness*minmax:
                champion = challenger
        new.append(champion)
    return new
