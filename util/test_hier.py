import sys
import os

import numpy

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import mobilib.region

# rels = mobilib.region.Relations(numpy.array([
    # [30, 3, 4, 2, 1, 0, 1, 1],
    # [ 6,15,10, 3, 2, 1, 0, 1],
    # [ 7, 9,12, 3, 1, 2, 0, 0],
    # [10, 3, 2,10, 0, 1, 2, 1],
    # [ 5, 8, 3, 1, 9, 4, 0, 0],
    # [ 3, 3, 7, 1, 2, 8, 1, 0],
    # [ 4, 0, 1, 6, 0, 1,10, 2],
    # [ 3, 1, 1, 5, 0, 1, 1, 6],
# ]))
rels = mobilib.region.Relations(numpy.array([
    [30, 3, 4, 2, 1, 0, 1, 1, 0],
    [ 6,15,10, 3, 2, 1, 0, 1, 0],
    [ 7, 9,12, 3, 1, 2, 0, 0, 0],
    [10, 3, 2,10, 0, 1, 2, 1, 0],
    [ 5, 8, 3, 1, 9, 4, 0, 0, 0],
    [ 3, 3, 7, 1, 2, 8, 1, 0, 0],
    [ 4, 0, 1, 6, 0, 1,10, 2, 0],
    [ 3, 1, 1, 5, 0, 1, 1, 6, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0],
]))
# parents = numpy.array((0,0,1,0,1,2,3,3))
# organ = numpy.array((0,0,1,0,0,0,0,0)).astype(bool)
# parents = numpy.array((0,0,1,1,1,1,1,1))
# organ = numpy.array((0,0,1,1,0,0,0,0)).astype(bool)
# hier2 = mobilib.region.Hierarchy.create(parents, organ)
# hierarchy = mobilib.region.MaxflowHierarchyBuilder().build(rels)
# hier2 = hierarchy.copy()
# hierarchy = mobilib.region.MaxflowStochasticHierarchyBuilder().build(rels)
criterion = mobilib.region.TransitionCriterion(organic_tolerance=1)

# print((numpy.where(numpy.isfinite(hierarchy.binding_matrix(rels)), hierarchy.binding_matrix(rels), 0) * 100).astype(int))
# print(hierarchy.structure_string())
# print(criterion.evaluate(rels, hierarchy))
# print()
# print()

# modifs = [cls() for cls in mobilib.region.HIERARCHY_MODIFIERS]
# n_modifs = 0
# while n_modifs < 10:
    # modif = numpy.random.choice(modifs, 1)[0]
    # success = modif.modify(hierarchy, rels)
    # if success:
        # n_modifs += 1
    # print(hierarchy.structure_string())
    # print(hierarchy.elements_by_id)
    # print(modif)
    # print((numpy.where(numpy.isfinite(hierarchy.binding_matrix(rels)), hierarchy.binding_matrix(rels), 0) * 100).astype(int))
    # print(hierarchy.structure_string())
    # print(hierarchy.elements_by_id)
    # print(criterion.evaluate(rels, hierarchy))
    # x = input()
    # print()
    
# cross = mobilib.region.HierarchyCrossover()
# print(hierarchy.structure_string())
# print()
# print(hier2.structure_string())
# print()
# # print(criterion.evaluate(rels, hierarchy))
# # print(hier2.structure_string())
# cross = cross.crossover(hierarchy, hier2, rels)
numpy.seterr(all='raise')
# genetic = mobilib.region.GeneticHierarchyBuilder()
determ = mobilib.region.MaxflowHierarchyBuilder()
# hierarchy = genetic.build(rels)
hierarchy = determ.build(rels)

print(hierarchy.structure_string())
print((numpy.where(numpy.isfinite(hierarchy.binding_matrix(rels)), hierarchy.binding_matrix(rels), 0) * 100).astype(int))
print(criterion.evaluate(rels, hierarchy))

# print(hier2.structure_string())
# print(criterion.evaluate(rels, hier2))

# hierarchy = 
