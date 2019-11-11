
import numpy as np
import scipy.sparse
import scipy.sparse.csgraph


relmatrix = np.array([
    [30, 3, 4, 2, 1, 0, 1, 1, 0, 0],
    [ 6,15,10, 3, 2, 1, 0, 1, 0, 0],
    [ 7, 9,12, 3, 1, 2, 0, 0, 0, 0],
    [10, 3, 2,10, 0, 1, 2, 1, 0, 0],
    [ 5, 8, 3, 1, 9, 4, 0, 0, 0, 0],
    [ 3, 3, 7, 1, 2, 8, 1, 0, 0, 0],
    [ 4, 0, 1, 6, 0, 1,10, 2, 0, 0],
    [ 3, 1, 1, 5, 0, 1, 1, 6, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 8, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 2, 4],
])

def delimit_hierarchy(relmatrix, weights=None):


    # will have to make a copy of rels and weights!
    relmatrix -= np.diag(np.diag(relmatrix))
    parents = np.empty_like(weights, dtype=np.int64)
    organicity = np.zeros_like(parents, dtype=bool)

    for comp_i in np.unique(compmem):
        in_comp = compmem == comp_i
        comprels = relmatrix[in_comp,:][:,in_comp]
        compweights = weights[in_comp]
        n_units = compweights.size
        comporgs = np.zeros_like(compweights, dtype=bool)
        while True:
            root_i = compweights.argmax()
            targets = comprels.argmax(axis=1)
            targets[root_i] = root_i
            # find strongly connected_components, these will be cycles
            target_neigh = scipy.sparse.csr_matrix((
                np.ones(n_units, dtype=bool),
                (np.arange(n_units), targets),
                ), shape=(n_units, n_units)
            )
            n_strong, strongcomps = scipy.sparse.csgraph.connected_components(
                target_neigh, directed=True, connection='strong', return_labels=True
            )
            if n_strong == n_units:
                break
            cyclecomps = np.flatnonzero(np.bincount(strongcomps) > 1)
            cycle_mem = np.where(
                np.isin(strongcomps, cyclecomps),
                strongcomps,
                -1
            )
            max_cycle_i = max(cyclecomps,
                key=lambda comp: compweights[cycle_mem == comp].sum()
            )
            max_cycle_mem = (cycle_mem == max_cycle_i)
            # contract the system
            main_i = (compweights * max_cycle_mem).argmax()
            all_is = np.flatnonzero(max_cycle_mem)
            other_is = np.array([i for i in all_is if i != main_i])
            comporgs[other_is] = main_i
            # contract weights
            compweights[main_i] += compweights[other_is].sum()
            compweights[other_is] = 0
            # contract relations
            comprels[:,main_i] += comprels[:,other_is].sum(axis=1)
            comprels[main_i,:] += comprels[other_is,:].sum(axis=0)
            comprels[:,other_is] = 0
            comprels[other_is,:] = 0
            comprels[other_is,main_i] = 1  # to ensure stable binding into the system
            comprels[main_i,main_i] = 0
        in_comp_is = np.flatnonzero(in_comp)
        parents[in_comp_is] = in_comp_is[targets]
        organicity[in_comp_is[comporgs]] = True
    return parents, organicity

print(parents)
print(organicity)