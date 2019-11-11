import argparse

import numpy
import pandas as pd

import mobilib.hierarchy
import mobilib.argparser

parser = mobilib.argparser.default(interactions=True)
parser.add_argument('out_file', help='file to write the output OD result')
parser.add_argument('-p', '--place-file',
    help='file with further information about places')
parser.add_argument('-i', '--place-id-col',
    help='field in place file with the place identifier matching the relation file')
parser.add_argument('-w', '--save-weights', action='store_true',
    help='save a weight column from the relations')
parser.add_argument('-b', '--save-bindings', action='store_true',
    help='save a hierarchy binding strength column from the relations')


if __name__ == '__main__':
    args = parser.parse_args()
    reldf = pd.read_csv(args.inter_file, sep=';')
    from_col, to_col, strength_col = reldf.columns[:3]
    if args.from_col: from_col = args.from_col
    if args.to_col: to_col = args.to_col
    if args.strength_col: strength_col = args.strength_col
    rels, ids = mobilib.hierarchy.Relations.from_dataframe(
        reldf, from_col, to_col, strength_col
    )
    # print(rels.matrix.shape, len(ids))
    # print(ids)
    # recs = []
    # for i, id in enumerate(ids):
        # irels = rels.matrix[i].copy()
        # itorels = rels.matrix[:,i].copy()
        # myrel = irels[i]
        # irels[i] = 0
        # itorels[i] = 0
        # recs.append((id, irels.max() / myrel, itorels.max() / myrel, irels.max(), itorels.max(), myrel, 1 - 2 * myrel / (irels.sum() + itorels.sum() + 2 * myrel)))
        # # print(list(sorted(zip(ids, ), key=lambda it: it[1])))
    # pd.DataFrame.from_records(recs, columns=['id', 'rat', 'torat', 'maxflow', 'maxtoflow', 'selfflow', 'flowrel']).to_csv(args.out_file, sep=';', index=False)
    builder = mobilib.hierarchy.NewMaxflowHierarchyBuilder()
    # builder = mobilib.hierarchy.MaxflowHierarchyBuilder()
    # builder = mobilib.region.GeneticHierarchyBuilder()
    hierarchy = builder.build(rels, ids=ids)
    # print(hierarchy.structure_string())
    
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
    if args.save_bindings:
        criterion = mobilib.hierarchy.TransitionCriterion()
        outdf['binding'] = criterion.evaluate_nodes(hierarchy, rels) ** 2
    if args.place_file:
        outdf = pd.merge(
            pd.read_csv(args.place_file, sep=';'),
            outdf,
            how='outer',
            left_on=args.place_id_col,
            right_on='id',
        )
    outdf.to_csv(args.out_file, sep=';', index=False)
