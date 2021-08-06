import itertools

#from bloopy.individual import individual, continuous individual
from bloopy.individual import individual
import bloopy.utils as utils

def get_variable_ranges(bound_list):
    var_ranges = []
    for var in range(len(bound_list)):
        rang = []
        for i in range(bound_list[var][0], bound_list[var][1]+1):
            rang.append(i)
        var_ranges.append(rang)
    return var_ranges

def generate_Hamming_neighbours(x0, varranges):
    neighbours = []
    for var in range(len(x0)):
        left = x0[:var]
        right = x0[var+1:]
        for y in varranges[var]:
            if y == x0[var]:
                continue
            n = left + (y,) + right
            neighbours.append(n)
    return neighbours

def generate_circular_neighbours(x0, varranges):
    neighbours = []
    for var in range(len(x0)):
        left = x0[:var]
        right = x0[var+1:]
        if x0[var] - 1 >= varranges[var][0]:
            y = x0[var] - 1
        else:
            y = varranges[var][-1]
        n1 = left + (y,) + right
        if x0[var] + 1 <= varranges[var][-1]:
            y = x0[var] + 1
        else:
            y = varranges[var][0]
        n2 = left + (y,) + right
        neighbours.append(n1)
        neighbours.append(n2)
    return neighbours

def generate_bounded_neighbours(x0, varranges):
    neighbours = []
    for var in range(len(x0)):
        n1 = None
        n2 = None
        left = x0[:var]
        right = x0[var+1:]
        if x0[var] - 1 >= varranges[var][0]:
            y = x0[var] - 1
            n1 = left + (y,) + right
        if x0[var] + 1 <= varranges[var][-1]:
            y = x0[var] + 1
            n2 = left + (y,) + right
        if n1 is not None:
            neighbours.append(n1)
        if n2 is not None:
            neighbours.append(n2)
    return neighbours

def generate_circular_neighbours_perdim(x0, varranges):
    neighbours = []
    for var in range(len(x0)):
        left = x0[:var]
        right = x0[var+1:]
        if x0[var] - 1 >= varranges[var][0]:
            y = x0[var] - 1
        else:
            y = varranges[var][-1]
        n1 = left + (y,) + right
        if x0[var] + 1 <= varranges[var][-1]:
            y = x0[var] + 1
        else:
            y = varranges[var][0]
        n2 = left + (y,) + right
        neighbours.append([n1, n2])
    return neighbours

def generate_bounded_neighbours_perdim(x0, varranges):
    neighbours = []
    for var in range(len(x0)):
        n1 = None
        n2 = None
        left = x0[:var]
        right = x0[var+1:]
        if x0[var] - 1 >= varranges[var][0]:
            y = x0[var] - 1
            n1 = left + (y,) + right
        if x0[var] + 1 <= varranges[var][-1]:
            y = x0[var] + 1
            n2 = left + (y,) + right
        lst = []
        if n1 is not None:
            lst.append(n1)
        if n2 is not None:
            lst.append(n2)
        neighbours.append(lst)
    return neighbours

def generate_Hamming_neighbours_perdim(x0, varranges):
    neighbours = []
    for var in range(len(x0)):
        left = x0[:var]
        right = x0[var+1:]
        lst = []
        for y in varranges[var]:
            if y == x0[var]:
                continue
            n = left + (y,) + right
            lst.append(n)
        neighbours.append(lst)
    return neighbours

def build_nodeidxs_dict(bound_list, fitfunc, bsize):
    #Give all point node indices for the graph, so an entry
    # is [point_type, fitness, node_index]
    indiv = individual(bsize, boundary_list=bound_list)
    var_ranges = get_variable_ranges(bound_list)
    node_dict = dict()
    node_idx = 0
    for x in itertools.product(*var_ranges):
        utils.set_bitstring(indiv, list(x))
        xfit = fitfunc(indiv.bitstring)
        node_dict[tuple(x)] = [node_idx, xfit]
        node_idx += 1
    return node_dict

def indices_to_points(node_dict):
    # We need a mapping from node-indices to points
    idxs_to_point = dict()
    for point, value in node_dict.items():
        # Save [point, fitness, ptype]
        idxs_to_point[value[2]] = [point, value[1], value[0]]
    return idxs_to_point
