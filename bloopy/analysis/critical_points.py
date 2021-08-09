import itertools
import warnings
import numpy as np

import bloopy.analysis.analysis_utils as anutil
from bloopy.individual import individual
import bloopy.utils as utils


def classify_points(bsize, bound_list, nidxs_dict, method='bounded'):
    if method not in ["circular", "bounded", "Hamming"]:
        raise Exception("Unknown neighbour generator given!")
    var_ranges = anutil.get_variable_ranges(bound_list)

    count_saddles = 0
    count_minima = 0
    count_maxima = 0
    regular_points = 0
    total = 0

    #A point is classified as type 0,1,2,3
    # 0 is regular, 1 is local minimum,
    # 2 is local maximum, and 3 is saddlepoint
    # we save points as tuple (type, fit, node_index) for convenience
    point_class = dict()
    for x in itertools.product(*var_ranges):
        nr_dims_is_minimal = 0
        isRegular = False
        xidx, xfit = nidxs_dict[x]
        if method == "circular":
            nbours = anutil.generate_circular_neighbours_perdim(x, var_ranges)
        elif method == "Hamming":
            nbours = anutil.generate_Hamming_neighbours_perdim(x, var_ranges)
        else:
            nbours = anutil.generate_bounded_neighbours_perdim(x, var_ranges)
        for dim in range(len(nbours)):
            nbours_larger = 0
            if len(nbours[dim]) == 0:
                raise Exception("Should already have been removed when processing space.")
            elif len(nbours[dim]) == 1:# Only one neighbour
                n1 = nbours[dim][0]
                n1_idxs, n1fit = nidxs_dict[n1]
                if n1fit == xfit:#This is for the failfit
                    if xfit > 10000:
                        isRegular = True
                        break
                    else:
                        print(xfit, "1")
                        warnings.warn("Equal fitness found")
                        #raise Exception("Equal fitness found")
                if n1fit >= xfit:
                    nbours_larger += 1
                    nr_dims_is_minimal += 1
            elif len(nbours[dim]) == 2:# Only one neighbour
                n1, n2 = nbours[dim]
                n1_idxs, n1fit = nidxs_dict[n1]
                n2_idxs, n2fit = nidxs_dict[n2]
                if n1fit == xfit or n2fit == xfit:#This is for the failfit
                    if xfit > 10000:
                        isRegular = True
                        break
                    else:
                        print(xfit, "2")
                        warnings.warn("Equal fitness found")
                        #raise Exception("Equal fitness found")
                if n1fit >= xfit:
                    nbours_larger += 1
                if n2fit >= xfit:
                    nbours_larger += 1
                if nbours_larger == 1:#Its not a critical point as it has
                    # a non-zero gradient in this dimension
                    isRegular = True
                    break
                elif nbours_larger == 2:
                    nr_dims_is_minimal += 1
            else:# More neighbours
                nbours_larger = 0
                for ni in nbours[dim]:
                    ni_idxs, nifit = nidxs_dict[ni]
                    if nifit == xfit:#This is for the failfit
                        if xfit > 10000:
                            isRegular = True
                            break
                        else:
                            warnings.warn("Equal fitness found")
                            print(xfit, "N")
                            #raise Exception("Equal fitness found")
                    if nifit >= xfit:
                        nbours_larger += 1
                if nbours_larger == len(nbours[dim]):#minimal in this dim
                    nr_dims_is_minimal += 1
                elif nbours_larger == 0:#maximal in this dim
                    pass
                else:#Its not a critical point as it has
                    # a non-zero gradient in this dimension
                    isRegular = True
                    break

        total += 1
        if x in point_class.keys():
            raise Exception("SOMETHING WRONG")

        if isRegular:
            regular_points += 1
            point_class[x] = [0, xfit, xidx]
            continue

        if nr_dims_is_minimal == 0:
            count_maxima += 1
            point_class[x] = [2, xfit, xidx]
        elif nr_dims_is_minimal == len(bound_list):
            count_minima += 1
            point_class[x] = [1, xfit, xidx]
        else:
            count_saddles += 1
            point_class[x] = [3, xfit, xidx]
    return total, count_minima, count_maxima, count_saddles, regular_points, point_class

def strong_local_minima(p, globfit, space_dict):
    loc_min_idxs = []
    for pt, value in space_dict.items():
        ptype, pfit, pidx = value
        if ptype != 1:
            continue
        if pfit <= (1+p)*globfit:
            loc_min_idxs.append(pidx)
    return loc_min_idxs

def strong_critical_points(p, globfit, space_dict):
    loc_min_idxs = []
    for pt, value in space_dict.items():
        ptype, pfit, pidx = value
        if ptype == 0:
            continue
        if pfit <= (1+p)*globfit:
            loc_min_idxs.append(pidx)
    return loc_min_idxs

def sizes_minima(sspace, bsize, bound_list, fitfunc, method="bounded"):
    if method not in ["circular", "bounded", "Hamming"]:
        raise Exception("Unknown neighbour generator given!")
    indiv = individual(bsize, boundary_list=bound_list)
    var_ranges = anutil.get_variable_ranges(bound_list)
    count_minima = 0
    total = 0
    hole_depths = []
    for x in itertools.product(*var_ranges):
        utils.set_bitstring(indiv, list(x))
        xfit = fitfunc(indiv.bitstring)

        if method == "circular":
            nbours = anutil.generate_circular_neighbours(x, var_ranges)
        elif method == "Hamming":
            nbours = anutil.generate_Hamming_neighbours(x, var_ranges)
        else:
            nbours = anutil.generate_bounded_neighbours(x, var_ranges)

        if xfit > 10000:
            is_minimal = False
        else:
            is_minimal = True
            nfits = []
            for nbour in nbours:
                utils.set_bitstring(indiv, nbour)
                nfit = fitfunc(indiv.bitstring)
                nfits.append(nfit)
                if nfit < xfit:# Neighbour has lower fitness
                    is_minimal = False
            if is_minimal:
                hole_depth = min(nfits) - xfit
                hole_depths.append([xfit, hole_depth])
                count_minima += 1
        total += 1
    return count_minima, total, np.array(hole_depths)
