
import numpy


def shortest_distances_multiplicative(adj):
    # floyd warshall, but with multiplication inside
    lengths = adj.copy()
    for k in range(adj.shape[0]):
        lengths = numpy.minimum(
            lengths,
            lengths[:,k][:,numpy.newaxis] * lengths[k,:][numpy.newaxis,:]
        )
    return lengths

    
def shortest_distances_additive(adj):
    # floyd warshall, but with multiplication inside
    lengths = adj.copy()
    for k in range(adj.shape[0]):
        lengths = numpy.minimum(
            lengths,
            lengths[:,k][:,numpy.newaxis] + lengths[k,:][numpy.newaxis,:]
        )
    return lengths

    
def parental_cycles(parents):
    n = parents.size
    visited = numpy.zeros(n, dtype=bool)
    n_vis = 0
    current = None
    while n_vis < n:
        if current is None:
            current = visited.argmin()
            path = []
        else:
            current = parents[current]
            n_vis += 1
        if current in path: # we found a cycle
            yield path[path.index(current):]
            current = None
        elif visited[current]: # we ran into an explored portion
            current = None
        else:
            path.append(current)
            visited[current] = True
