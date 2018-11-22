import argparse

import numpy
import pandas as pd

import mobilib.region

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('relation_file', help='file with OD relations between places')
   
    
if __name__ == '__main__':
    args = parser.parse_args()
    df = pd.read_csv(args.relation_file, sep=';')
    rels, ids = mobilib.region.Relations.from_dataframe(df, *df.columns[:3])
    builder = mobilib.region.MaxflowStochasticHierarchyBuilder()
    hierarchy = builder.build(rels, ids=ids)
    # print(hierarchy.structure_string())
    criterion = mobilib.region.TransitionCriterion(organic_tolerance=2)
    print(criterion.evaluate(rels, hierarchy))
    mover = mobilib.region.RootMover()
    mover.modify(hierarchy, rels)
    # print(hierarchy.tree(ids=ids).structure_string())