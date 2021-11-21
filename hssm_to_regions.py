"""Aggregate a hierarchical settlement system model to regions with a minimum weight threshold."""

import pandas as pd

import mobilib.argparser
import mobilib.hssm

parser = mobilib.argparser.default(__doc__)
parser.add_argument(
    'hssm_file',
    help='semicolon-delimited CSV file with the hierarchical settlement system model definition'
)
parser.add_argument(
    'out_file',
    help='file to write the output regional assignment'
)
parser.add_argument(
    '-i', '--id-col', default='id',
    help='unit ID column in the HSSM file'
)
parser.add_argument(
    '-p', '--parent-col', default='parent_id',
    help='parent unit ID column in the HSSM file'
)
parser.add_argument(
    '-s', '--stage-col', default='stage',
    help='unit settlement system stage column in the HSSM file'
)
parser.add_argument(
    '-w', '--weight-col', default='weight',
    help='unit weight column in the HSSM file'
)
parser.add_argument(
    '-r', '--region-col', default='region',
    help='name of the aggregated region column in the output file'
)
parser.add_argument(
    '-m', '--min-weight', default=None, type=float,
    help='minimum weight of region; if not set, determine the optimal one'
         ' from data by finding the most significant gap'
)
parser.add_argument(
    '-o', '--min-weight-sel-opts', default=0, type=int,
    help='let the user select the optimal weight from this many options;'
         ' only has effect when -m is not set'
)


if __name__ == '__main__':
    args = parser.parse_args()
    hssm_df = pd.read_csv(args.hssm_file, sep=';')
    ids = hssm_df[args.id_col]
    weights = hssm_df[args.weight_col].values
    hssm = mobilib.hssm.Model.from_arrays(
        parents=hssm_df[args.parent_col],
        stages=hssm_df[args.stage_col],
        index=ids,
    )
    raise NotImplementedError
    crclass = mobilib.hssm.WeightAggregationCriterion
    if args.min_weight:
        criterion = crclass(minimum=args.min_weight)
    else:
        criterion = crclass.optimal(hssm, weights, select=args.min_weight_sel_opts)
        if not args.min_weight_sel_opts:
            print('Optimal weight threshold determined at {:n}'.format(criterion.minimum))
    regsys = criterion.aggregate(hssm, weights)
    hssm_df[args.region_col] = ids[regsys.to_array()].values
    hssm_df.to_csv(args.out_file, sep=';', index=False)
