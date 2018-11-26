import argparse

import numpy
import pandas as pd

import mobilib.hierarchy

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('relation_file', help='file with OD relations between places')
parser.add_argument('out_file', help='file to write the output OD result')
parser.add_argument('-p', '--place-file',
    help='file with further information about places')
parser.add_argument('-i', '--place-id-col',
    help='field in place file with the place identifier matching the relation file')
parser.add_argument('-f', '--from-col',
    help='field in relation table containing the relation source place identifier')
parser.add_argument('-t', '--to-col',
    help='field in relation table containing the target place identifier')
parser.add_argument('-s', '--strength-col',
    help='field in relation table containing the relation strength')
parser.add_argument('-w', '--save-weights', action='store_true',
    help='save a weight column from the relations')
   
    
if __name__ == '__main__':
    args = parser.parse_args()
    reldf = pd.read_csv(args.relation_file, sep=';')
    from_col, to_col, strength_col = reldf.columns[:3]
    if args.from_col: from_col = args.from_col
    if args.to_col: to_col = args.to_col
    if args.strength_col: strength_col = args.strength_col
    rels, ids = mobilib.hierarchy.Relations.from_dataframe(reldf, from_col, to_col, strength_col)
    builder = mobilib.hierarchy.MaxflowHierarchyBuilder()
    # builder = mobilib.region.GeneticHierarchyBuilder()
    hierarchy = builder.build(rels, ids=ids)
    # print(hierarchy.structure_string())
    criterion = mobilib.hierarchy.TransitionCriterion()
    # mover = mobilib.hierarchy.RootMover()
    # mover.modify(hierarchy, rels)
    parents, organics = hierarchy.to_arrays()
    outdf = pd.DataFrame({
        'id' : ids,
        'parent_id' : [ids[parent] for parent in parents],
        'organ' : organics
    })
    if args.save_weights:
        outdf['weight'] = rels.weights
    if args.place_file:
        outdf = pd.merge(
            pd.read_csv(args.place_file, sep=';'),
            outdf,
            how='outer',
            left_on=args.place_id_col,
            right_on='id',
        )
    outdf.to_csv(args.out_file, sep=';', index=False)
