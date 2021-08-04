import random
from timeit import default_timer as timer
from bitarray import bitarray
import numpy as np
import copy

from bloopy.individual import individual

def RandomGreedyHillclimb(candidate, ffunc, minmax, totfevals, maxfeval, visited_cache, nbour_method, restart=True):
    if nbour_method not in ["Hamming", "adjacent"]:
        raise Exception("Unknown neighbour method.")
    splits = candidate.boundary_list
    foundimprove = True
    func_evals = 0
    total_fit = 0
    bs_size = candidate.size
    while foundimprove:
        child = individual(candidate.size, bitstring=copy.deepcopy(candidate.bitstring), boundary_list=splits)
        child.fitness = candidate.fitness
        foundimprove = False
        if splits is None:
            # Do nothing with neighbour method for bitstrings
            # We need this to randomly go through the neighbours
            shuffle = np.random.permutation(bs_size)
            for k in range(bs_size):
                if maxfeval is not None and totfevals + func_evals >= maxfeval:
                    break
                idx = shuffle[k]
                child.bitstring[idx] = not child.bitstring[idx]
                bsstr = child.bitstring.to01()
                if bsstr in visited_cache:
                    child.fitness = visited_cache[bsstr]
                else:
                    child.fitness = ffunc(child.bitstring)
                    visited_cache[bsstr] = child.fitness
                    func_evals += 1
                    total_fit += child.fitness
                if minmax * child.fitness > minmax * candidate.fitness:
                    candidate = copy.deepcopy(child)
                    foundimprove = True
                    if restart:
                        break
                else:
                    child.bitstring[idx] = not child.bitstring[idx] # flip back
                    child.fitness = candidate.fitness
        else:
            indices = [i for i, x in enumerate(list(child.bitstring)) if x]
            shuffle_pars = np.random.permutation(len(splits)).tolist()
            for k in shuffle_pars:
                if nbour_method == "Hamming":
                    bs_idxs = np.arange(splits[k][0], splits[k][1]+1)
                    bs_idxs = np.random.permutation(bs_idxs).tolist()
                elif nbour_method == "adjacent":
                    if indices[k] == splits[k][0]:
                        bs_idxs = [indices[k]+1]
                    elif indices[k] == splits[k][1]:
                        bs_idxs = [indices[k]-1]
                    else:
                        bs_idxs = [indices[k]-1, indices[k]+1]
                for i in bs_idxs:
                    if i == indices[k]:
                        continue
                    if maxfeval is not None and totfevals + func_evals >= maxfeval:
                        break
                    child.bitstring[indices[k]] = 0 # Set old one to 0
                    child.bitstring[i] = 1 # set new one to 1
                    bsstr = child.bitstring.to01()
                    if bsstr in visited_cache:
                        child.fitness = visited_cache[bsstr]
                    else:
                        child.fitness = ffunc(child.bitstring)
                        visited_cache[bsstr] = child.fitness
                        func_evals += 1
                        total_fit += child.fitness
                    if minmax * child.fitness > minmax * candidate.fitness:
                        candidate = copy.deepcopy(child)
                        indices[k] = i
                        foundimprove = True
                        if restart:
                            break
                    else:
                        child.bitstring[i] = 0 # set new one to 0
                        child.bitstring[indices[k]] = 1 # Set old one back to 1
                        child.fitness = candidate.fitness
                if foundimprove and restart:
                    break
    return candidate, func_evals, total_fit, visited_cache

def OrderedGreedyHillclimb(candidate, ffunc, minmax, totfevals, maxfeval, visited_cache, nbour_method, order, restart=True):
    if nbour_method not in ["Hamming", "adjacent"]:
        raise Exception("Unknown neighbour method.")
    splits = candidate.boundary_list
    foundimprove = True
    func_evals = 0
    total_fit = 0
    bs_size = candidate.size
    while foundimprove:
        child = individual(candidate.size, bitstring=copy.deepcopy(candidate.bitstring), boundary_list=splits)
        child.fitness = candidate.fitness
        foundimprove = False
        if splits is None:
            # Do nothing with neighbour method for bitstrings
            if order is None:
                bs_idxs = list(range(bs_size))
            else:
                bs_idxs = order
            for k in bs_idxs:
                if maxfeval is not None and totfevals + func_evals >= maxfeval:
                    break
                child.bitstring[k] = not child.bitstring[k]
                bsstr = child.bitstring.to01()
                if bsstr in visited_cache:
                    child.fitness = visited_cache[bsstr]
                else:
                    child.fitness = ffunc(child.bitstring)
                    visited_cache[bsstr] = child.fitness
                    func_evals += 1
                    total_fit += child.fitness
                if minmax * child.fitness > minmax * candidate.fitness:
                    candidate = copy.deepcopy(child)
                    foundimprove = True
                    if restart:
                        break
                else:
                    child.bitstring[k] = not child.bitstring[k] # flip back
                    child.fitness = candidate.fitness
        else:
            indices = [i for i, x in enumerate(list(child.bitstring)) if x]
            if order is None:
                var_idxs = list(range(len(splits)))
            else:
                var_idxs = order
            for k in var_idxs:
                if nbour_method == "Hamming":
                    bs_idxs = list(range(splits[k][0], splits[k][1]+1))
                elif nbour_method == "adjacent":
                    if indices[k] == splits[k][0]:
                        bs_idxs = [indices[k]+1]
                    elif indices[k] == splits[k][1]:
                        bs_idxs = [indices[k]-1]
                    else:
                        bs_idxs = [indices[k]-1, indices[k]+1]
                for i in bs_idxs:
                    if i == indices[k]:
                        continue
                    if maxfeval is not None and totfevals + func_evals >= maxfeval:
                        break
                    child.bitstring[indices[k]] = 0 # Set old one to 0
                    child.bitstring[i] = 1 # set new one to 1
                    bsstr = child.bitstring.to01()
                    if bsstr in visited_cache:
                        child.fitness = visited_cache[bsstr]
                    else:
                        child.fitness = ffunc(child.bitstring)
                        visited_cache[bsstr] = child.fitness
                        func_evals += 1
                        total_fit += child.fitness
                    if minmax * child.fitness > minmax * candidate.fitness:
                        candidate = copy.deepcopy(child)
                        indices[k] = i
                        foundimprove = True
                        if restart:
                            break
                    else:
                        child.bitstring[i] = 0 # set new one to 0
                        child.bitstring[indices[k]] = 1 # Set old one back to 1
                        child.fitness = candidate.fitness
                if foundimprove and restart:
                    break
    return candidate, func_evals, total_fit, visited_cache

def BestHillclimb(candidate, ffunc, minmax, totfevals, maxfeval, visited_cache, nbour_method):
    if nbour_method not in ["Hamming", "adjacent"]:
        raise Exception("Unknown neighbour method.")
    splits = candidate.boundary_list
    foundimprove = True
    func_evals = 0
    total_fit = 0
    bs_size = candidate.size
    while foundimprove:
        child = individual(candidate.size, bitstring=copy.deepcopy(candidate.bitstring), boundary_list=splits)
        best_child = individual(candidate.size, bitstring=copy.deepcopy(candidate.bitstring), boundary_list=splits)
        child.fitness = candidate.fitness
        best_child.fitness = candidate.fitness
        foundimprove = False
        if splits is None:
            # Do nothing with neighbour method for bitstrings
            for k in range(bs_size):
                if maxfeval is not None and totfevals + func_evals >= maxfeval:
                    break
                child.bitstring[k] = not child.bitstring[k]
                bsstr = child.bitstring.to01()
                if bsstr in visited_cache:
                    child.fitness = visited_cache[bsstr]
                else:
                    child.fitness = ffunc(child.bitstring)
                    visited_cache[bsstr] = child.fitness
                    func_evals += 1
                    total_fit += child.fitness

                #If a neighbour is better, than the best found neighbours, save it
                if minmax * child.fitness > minmax * best_child.fitness:
                    best_child = copy.deepcopy(child)

                # Return to the original child
                child.bitstring[k] = not child.bitstring[k] # flip back
                child.fitness = candidate.fitness
            #If the best neighbour is an improvement, move there and repeat
            if minmax * best_child.fitness > minmax * candidate.fitness:
                foundimprove = True
                candidate = copy.deepcopy(best_child)
        else:
            indices = [i for i, x in enumerate(list(child.bitstring)) if x]
            # Go through all neighbours with one of the copies
            for k in range(len(splits)):
                if nbour_method == "Hamming":
                    bs_idxs = list(range(splits[k][0], splits[k][1]+1))
                elif nbour_method == "adjacent":
                    if indices[k] == splits[k][0]:
                        bs_idxs = [indices[k]+1]
                    elif indices[k] == splits[k][1]:
                        bs_idxs = [indices[k]-1]
                    else:
                        bs_idxs = [indices[k]-1, indices[k]+1]
                for i in bs_idxs:
                    if i == indices[k]:
                        continue
                    if maxfeval is not None and totfevals + func_evals >= maxfeval:
                        break
                    child.bitstring[indices[k]] = 0 # Set old one to 0
                    child.bitstring[i] = 1 # set new one to 1
                    bsstr = child.bitstring.to01()
                    if bsstr in visited_cache:
                        child.fitness = visited_cache[bsstr]
                    else:
                        child.fitness = ffunc(child.bitstring)
                        visited_cache[bsstr] = child.fitness
                        func_evals += 1
                        total_fit += child.fitness

                    #If a neighbour is better, than the best found neighbours, save it
                    if minmax * child.fitness > minmax * best_child.fitness:
                        best_child = copy.deepcopy(child)

                    # Return to the original child
                    child.bitstring[i] = 0 # set new one to 0
                    child.bitstring[indices[k]] = 1 # Set old one back to 1
                    child.fitness = candidate.fitness
            #If the best neighbour is an improvement, move there and repeat
            if minmax * best_child.fitness > minmax * candidate.fitness:
                foundimprove = True
                candidate = copy.deepcopy(best_child)
    return candidate, func_evals, total_fit, visited_cache

def StochasticHillclimb(candidate, ffunc, minmax, totfevals, maxfeval, visited_cache, nbour_method):
    if nbour_method not in ["Hamming", "adjacent"]:
        raise Exception("Unknown neighbour method.")
    splits = candidate.boundary_list
    foundimprove = True
    func_evals = 0
    total_fit = 0
    bs_size = candidate.size
    while foundimprove:
        if maxfeval is not None and totfevals >= maxfeval:
            break
        uphill_moves = []
        child = individual(candidate.size, bitstring=copy.deepcopy(candidate.bitstring), boundary_list=candidate.boundary_list)
        child.fitness = candidate.fitness
        foundimprove = False
        if splits is None:
            # Do nothing with neighbour method for bitstrings
            for k in range(bs_size):
                if maxfeval is not None and totfevals + func_evals >= maxfeval:
                    break
                child.bitstring[k] = not child.bitstring[k]
                bsstr = child.bitstring.to01()
                if bsstr in visited_cache:
                    child.fitness = visited_cache[bsstr]
                else:
                    child.fitness = ffunc(child.bitstring)
                    visited_cache[bsstr] = child.fitness
                    func_evals += 1
                    total_fit += child.fitness

                if minmax * child.fitness > minmax * candidate.fitness:
                    improved_candidate = copy.deepcopy(child)
                    uphill_moves.append((improved_candidate, abs(child.fitness-candidate.fitness)))
                    foundimprove = True
                child.bitstring[k] = not child.bitstring[k] # flip back
                child.fitness = candidate.fitness
        else:
            indices = [i for i, x in enumerate(list(child.bitstring)) if x]
            for k in range(len(splits)):
                if nbour_method == "Hamming":
                    bs_idxs = list(range(splits[k][0], splits[k][1]+1))
                elif nbour_method == "adjacent":
                    if indices[k] == splits[k][0]:
                        bs_idxs = [indices[k]+1]
                    elif indices[k] == splits[k][1]:
                        bs_idxs = [indices[k]-1]
                    else:
                        bs_idxs = [indices[k]-1, indices[k]+1]
                for i in bs_idxs:
                    if i == indices[k]:
                        continue
                    if maxfeval is not None and totfevals + func_evals >= maxfeval:
                        break
                    child.bitstring[indices[k]] = 0 # Set old one to 0
                    child.bitstring[i] = 1 # set new one to 1
                    bsstr = child.bitstring.to01()
                    if bsstr in visited_cache:
                        child.fitness = visited_cache[bsstr]
                    else:
                        child.fitness = ffunc(child.bitstring)
                        visited_cache[bsstr] = child.fitness
                        func_evals += 1
                        total_fit += child.fitness
                    if minmax * child.fitness > minmax * candidate.fitness:
                        improved_candidate = copy.deepcopy(child)
                        uphill_moves.append((improved_candidate, abs(child.fitness-candidate.fitness)))
                        foundimprove = True
                    child.bitstring[i] = 0 # set new one to 0
                    child.bitstring[indices[k]] = 1 # Set old one back to 1
                    child.fitness = candidate.fitness

        if len(uphill_moves) > 0:
            probs = [x[1] for x in uphill_moves]
            # Choose a random uphill move proportionate to fitness increase
            index = np.random.choice(range(len(uphill_moves)),size=(1,),p=(np.array(probs)/float(sum(probs))).tolist())[0]
            candidate = copy.deepcopy(uphill_moves[index][0])
    return candidate, func_evals, total_fit, visited_cache
