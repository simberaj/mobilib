import argparse

import numpy
import pandas as pd

import mobilib.hierarchy
import mobilib.argparser

parser = mobilib.argparser.default(__doc__, interactions=True)
parser.add_argument('out_file', help='file to write the output OD result')
parser.add_argument('-p', '--place-file',
    help='file with further information about places')
parser.add_argument('-i', '--place-id-col',
    help='field in place file with the place identifier matching the relation file')
parser.add_argument('-w', '--save-weights', action='store_true',
    help='save a weight column from the relations')
parser.add_argument('-b', '--save-bindings', action='store_true',
    help='save a hierarchy binding strength column from the relations')
parser.add_argument('-C', '--criterion', default='transition',
    help='the evaluation criterion to use to determine hierarchy binding strength')
parser.add_argument('-B', '--builder', default='maxflow',
    help='the hierarchy builder to use')


if __name__ == '__main__':
    args = parser.parse_args()
    reldf = pd.read_csv(args.inter_file, sep=';')
    from_col, to_col, strength_col = reldf.columns[:3]
    if args.from_id_col: from_col = args.from_id_col
    if args.to_id_col: to_col = args.to_id_col
    if args.strength_col: strength_col = args.strength_col
    rels, ids = mobilib.hierarchy.Relations.from_dataframe(
        reldf, from_col, to_col, strength_col
    )
    criterion = mobilib.hierarchy.criterion(args.criterion)
    builder = mobilib.hierarchy.builder(args.builder, criterion=criterion)
    hierarchy = builder.build(rels, ids=ids)
    parents, organics = hierarchy.to_arrays()
    outdf = pd.DataFrame({
        'id': ids,
        'parent_id': [ids[parent] for parent in parents],
        'organ': organics
    })
    if args.save_weights:
        outdf['weight'] = rels.weights
    if args.save_bindings:
        outdf['binding'] = criterion.evaluate_nodes(hierarchy, rels)
    if args.place_file:
        outdf = pd.merge(
            pd.read_csv(args.place_file, sep=';'),
            outdf,
            how='outer',
            left_on=args.place_id_col,
            right_on='id',
        )
    outdf.to_csv(args.out_file, sep=';', index=False)
