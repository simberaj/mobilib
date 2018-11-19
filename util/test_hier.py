import numpy

import mobilib.region

rels = mobilib.region.Relations(numpy.array([
    [30, 3, 4, 2, 1, 0, 1, 1],
    [ 6,15,10, 3, 2, 1, 0, 1],
    [ 7, 9,12, 3, 1, 2, 0, 0],
    [10, 3, 2,10, 0, 1, 2, 1],
    [ 5, 8, 3, 1, 9, 4, 0, 0],
    [ 3, 3, 7, 1, 2, 8, 1, 0],
    [ 4, 0, 1, 6, 0, 1,10, 2],
    [ 3, 1, 1, 5, 0, 1, 1, 6],
]))
parents = numpy.array((0,0,1,0,1,2,3,3))
organ = numpy.array((0,0,1,0,0,0,0,0)).astype(bool)
# parents = numpy.array((0,0,1,1,1,1,1,1))
# organ = numpy.array((0,0,1,1,0,0,0,0)).astype(bool)
hierarchy = mobilib.region.Hierarchy.create(parents, organ, rels=rels)
# hierarchy = mobilib.region.MaxflowStochasticHierarchyBuilder().build(rels)

print(rels.transition_probs)
print(rels.weights)

print(hierarchy.structure_string())
print((hierarchy.binding_matrix * 100).astype(int))

criterion = mobilib.region.TransitionCriterion(organic_tolerance=1)
print(criterion.evaluate(rels, hierarchy))

# print(tree.structure_string(rels.transition_probs[numpy.arange(rels.n),parents]))
