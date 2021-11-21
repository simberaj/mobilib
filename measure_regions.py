"""Compute measures for regions.

Computes loads of numerical measures for regions according to a given delimitation from the
properties of their units and their interactions.

TODO list default measures
"""

import pandas as pd

import mobilib.argparser
import mobilib.region_measure


parser = mobilib.argparser.default(__doc__, interactions=True)
parser.add_argument(
    'unit_file',
    help='semicolon-delimited CSV with unit data and mapping to regions'
)
parser.add_argument(
    'out_file',
    help='path to output file with regions and their measures'
)
parser.add_argument(
    '-u', '--unit-id-col', default='id',
    help='name of the unit ID attribute in the unit file'
)
parser.add_argument(
    '-p', '--unit-prop-col', nargs=2, action='append', default=[],
    metavar=('PROPERTY', 'COLUMN'),
    help='name of the unit property and its corresponding column in the unit'
         ' file to load and use'
)
parser.add_argument(
    '-r', '--unit-region-col', default='region',
    help='name of the region ID attribute in the unit file'
)
parser.add_argument(
    '-c', '--unit-core-col',
    help='name of the core indicator attribute in the unit file'
)
parser.add_argument(
    '-n', '--unit-name-col',
    help='name of the unit name attribute in the unit file; if given,'
         ' automatically includes the computation of the name measure'
)
parser.add_argument(
    '-m', '--measures', nargs='+',
    help='explicitly list measures to compute'
)
parser.add_argument(
    '-M', '--exclude-measures', nargs='+',
    help='explicitly list measures from the default list not to compute'
)
parser.add_argument(
    '-l', '--distinguish-largest-by-col', default='mass',
    help='distinguish largest unit of region by this property as specified in --unit-prop-col'
)
parser.add_argument(
    '-F', '--fraction-for-largest', default=1., type=float,
    help='make all units with largest-distinction property higher than this'
         ' fraction of the largest unit also largest'
)


if __name__ == '__main__':
    args = parser.parse_args()
    unit_df_raw = pd.read_csv(args.unit_file, sep=';', encoding='utf8')
    inter_df_raw = pd.read_csv(args.inter_file, sep=';')
    renamer = dict((column, prop) for prop, column in args.unit_prop_col)
    renamer[args.unit_region_col] = 'region'
    if args.unit_core_col:
        renamer[args.unit_core_col] = 'is_core'
    if args.unit_name_col:
        renamer[args.unit_name_col] = 'name'
    unit_df = unit_df_raw.set_index(args.unit_id_col)[list(renamer.keys())].rename(columns=renamer)
    inter_df = inter_df_raw.set_index([args.from_id_col, args.to_id_col])[[args.strength_col]]
    measure_df = mobilib.region_measure.calculate(
        unit_df, inter_df,
        only_measures=args.measures,
        exclude_measures=args.exclude_measures,
        largest_prop=args.distinguish_largest_by_col,
        largest_fraction=args.fraction_for_largest,
    )
    measure_df.reset_index().to_csv(args.out_file, sep=';', encoding='utf8', index=False)
