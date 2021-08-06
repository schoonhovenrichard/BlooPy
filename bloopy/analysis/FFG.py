import networkx as nx
import itertools

import bloopy.analysis.analysis_utils as anutil

def build_FFG(nidxs_dict, bound_list, method='bounded'):
    if method not in ["circular", "bounded", "Hamming"]:
        raise Exception("Unknown neighbour generator given!")
    var_ranges = anutil.get_variable_ranges(bound_list)

    G = nx.DiGraph()
    for key, value in nidxs_dict.items():
        nidx = value[0]
        if nidx not in G:
            G.add_node(nidx)

    it = 0
    for x in itertools.product(*var_ranges):
        it += 1
        print('Adding point to CPG, node {}...\r'.format(it), end="")
        xidx, xfit = nidxs_dict[x]
        if method == "circular":
            nbours = anutil.generate_circular_neighbours(x, var_ranges)
        elif method == "Hamming":
            nbours = anutil.generate_Hamming_neighbours(x, var_ranges)
        else:
            nbours = anutil.generate_bounded_neighbours(x, var_ranges)
        for nbour in nbours:
             nidx, nfit = nidxs_dict[nbour]
             if nfit <= xfit:
             #if nfit < xfit:
                 G.add_edge(xidx, nidx)
    return G

def average_centrality_nodes(centrality_dict, nodelist, space_dict, nidxs_pnts):
    nodelist_centrality = 0.0
    for node in nodelist:
        nodelist_centrality += abs(centrality_dict[node])
    minima_centralities = []
    for idx, centr in centrality_dict.items():
        pnt = nidxs_pnts[idx]
        #[point, fitness, ptype]
        if pnt[2] != 1:
            continue
        minima_centralities.append(abs(centr))
    total_centrality = sum([abs(n) for i, n in centrality_dict.items()])
    minima_centrality = sum(minima_centralities)
    return nodelist_centrality, total_centrality, minima_centrality

def average_centrality_crits(centrality_dict, nodelist, space_dict, nidxs_pnts):
    nodelist_centrality = 0.0
    for node in nodelist:
        nodelist_centrality += abs(centrality_dict[node])
    critical_centralities = []
    for idx, centr in centrality_dict.items():
        #[point, fitness, ptype]
        pnt = nidxs_pnts[idx]
        critical_centralities.append(abs(centr))
    total_centrality = sum([abs(n) for i, n in centrality_dict.items()])

