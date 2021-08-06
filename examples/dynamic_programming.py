import numpy as np
import fitness_functions as ff

def bruteforce_MK_solve(mk_func):
    r"""
    Bruteforce computation of MK function optimum.
    """
    best_fit = 0.0
    best_bitset = ff.create_bitstring(0, mk_func.m)
    for i in range(2**mk_func.m):
        bs = ff.create_bitstring(i, mk_func.m)
        fitness = mk_func.get_fitness(bs)
        if fitness > best_fit:
            best_fit = fitness
            best_bitset = bs
    return best_fit, best_bitset

def dp_solve_MK(mk_func):
    r"""
    Function to find optimal solution to adjecent MK function using 
     dynamic programming.
    """
    m = mk_func.m
    k = mk_func.k
    if (m % (k-1) != 0):
        raise Exception("We have not decided how to implement DP if there are bits left over")
    tilde_functions = []
    q = int(m / (k-1))
    for i in range(q):
        tilde_function = dict()
        start = (k-1) * i
        for a0 in range(2**(2*k-2)):
            tilde_val = 0
            temp_bs = ff.create_bitstring(a0, 2*k-2)
            for j in range(k-1):
                current_mk_func = mk_func.fitness_map[start + j]
                temp_bs1 = ff.create_bitstring(0, k)
                for p in range(k):
                    temp_bs1[p] = temp_bs[j+p]
                tilde_val += current_mk_func[tuple(temp_bs1)]
            tilde_function[tuple(temp_bs)] = tilde_val
        tilde_functions.append(tilde_function)
    return dp_solve_adjacentMK_tilde(tilde_functions, q-1, k)

def dp_solve_adjacentMK_tilde(tilde_funcs, pos, k):
    r"""
    Helper function to DP solve adjacent MK functions.
    """
    if pos == 1:
        best_val = 0
        for a1 in range(2**(k-1)):
            for a2 in range(2**(k-1)):
                bsa1 = ff.create_bitstring(a1, k-1)
                bsa2 = ff.create_bitstring(a2, k-1)
                tilde0_bs = ff.create_bitstring(0, 2*k-2)
                tilde1_bs = ff.create_bitstring(0, 2*k-2)
                for i in range(k-1):
                    tilde0_bs[i] = bsa1[i]
                    tilde1_bs[k-1+i] = bsa1[i]
                for i in range(k-1):
                    tilde0_bs[k-1+i] = bsa2[i]
                    tilde1_bs[i] = bsa2[i]
                val = tilde_funcs[0][tuple(tilde0_bs)] + tilde_funcs[1][tuple(tilde1_bs)]
                if val > best_val:
                    best_val = val
        return best_val

    new_tilde_funcs = []
    for i in range(pos-1):
        new_tilde_funcs.append(tilde_funcs[i])
    tilde1 = tilde_funcs[pos-1]
    tilde2 = tilde_funcs[pos]
    new_tilde = dict()
    for a1 in range(2**(k-1)):
        for a2 in range(2**(k-1)):
            bsa1 = ff.create_bitstring(a1, k-1)
            bsa2 = ff.create_bitstring(a2, k-1)
            new_tilde_inp = ff.create_bitstring(0, 2*k-2)
            for i in range(k-1):
                new_tilde_inp[i] = bsa1[i]
            for i in range(k-1):
                new_tilde_inp[k-1+i] = bsa2[i]
            bs_tilde1 = ff.create_bitstring(0, 2*k-2)
            bs_tilde2 = ff.create_bitstring(0, 2*k-2)
            for i in range(k-1):
                bs_tilde1[i] = bsa1[i]
            for i in range(k-1):
                bs_tilde2[k-1+i] = bsa2[i]
            best_bval = 0
            for b in range(2**(k-1)):
                bsb = ff.create_bitstring(b, k-1)
                for j in range(k-1):
                    bs_tilde1[k-1+j] = bsb[j]
                for j in range(k-1):
                    bs_tilde2[j] = bsb[j]
                bval = tilde1[tuple(bs_tilde1)] + tilde2[tuple(bs_tilde2)]
                if bval > best_bval:
                    best_bval = bval
            new_tilde[tuple(new_tilde_inp)] = best_bval
    new_tilde_funcs.append(new_tilde)
    return dp_solve_adjacentMK_tilde(new_tilde_funcs, pos-1, k)
