import argparse
import datetime

import numpy as np
import pandas as pd

import mobilib.hssm
import mobilib.argparser


def impl_conv(value):
    if value.isdigit():
        return int(value)
    elif all(c.isdigit() or c == '.' for c in value):
        return float(value)
    else:
        return value


parser = mobilib.argparser.default(__doc__, interactions=True)
parser.add_argument('out_file', help='file to write the output model table')
parser.add_argument('-p', '--place-file',
    help='file with further information about places')
parser.add_argument('-i', '--place-id-col',
    help='field in place file with the place identifier matching the relation file')
parser.add_argument('-w', '--save-weights', action='store_true',
    help='save a weight column from the relations')
parser.add_argument('-b', '--save-bindings', action='store_true',
    help='save a hierarchy binding strength column from the relations')
parser.add_argument('-C', '--criterion', default='mfpt',
    help='the evaluation criterion to use to determine hierarchy binding strength')
parser.add_argument('-B', '--builder', default='maxflow',
    help='the hierarchy builder to use')
parser.add_argument('-S', '--seed', type=int,
    help='seed for the builder random generator')
parser.add_argument('-P', '--param', nargs=2, action='append',
    help='parameters for the builder')


if __name__ == '__main__':
    args = parser.parse_args()
    reldf = pd.read_csv(args.inter_file, sep=';')
    from_col, to_col, strength_col = reldf.columns[:3]
    if args.from_id_col: from_col = args.from_id_col
    if args.to_id_col: to_col = args.to_id_col
    if args.strength_col: strength_col = args.strength_col
    rels, ids = mobilib.hssm.Relations.from_dataframe(
        reldf, from_col, to_col, strength_col
    )
    criterion = mobilib.hssm.fitness_criterion(args.criterion)
    builder_params = {k: impl_conv(v) for k, v in args.param} if args.param else {}
    builder = mobilib.hssm.model_builder(
        args.builder,
        criterion=criterion,
        **builder_params,
    )
    if args.seed:
        np.random.seed(args.seed)
    start_dt = datetime.datetime.today()
    model = builder.build(rels, index=ids)
    end_dt = datetime.datetime.today()
    print('model created with fitness', criterion.evaluate(model, rels))
    # outdf = model.to_df()
    outdf = model.df.copy().rename_axis(index='id').reset_index()
    print(len(outdf), 'units in', (end_dt - start_dt).total_seconds(), 's')
    if args.save_weights:
        outdf['weight'] = rels.weights
    if args.save_bindings:
        outdf['binding'] = criterion.evaluate_nodes(model, rels)
    if args.place_file:
        outdf = pd.merge(
            pd.read_csv(args.place_file, sep=';'),
            outdf,
            how='outer',
            left_on=args.place_id_col,
            right_on='id',
        )
    outdf.to_csv(args.out_file, sep=';', index=False)
