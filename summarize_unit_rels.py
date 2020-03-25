'''Summarize interactions by source unit and regional relation classification of target.'''

import argparse

import numpy as np
import pandas as pd

import mobilib.argparser

parser = mobilib.argparser.default(__doc__, interactions=True)
parser.add_argument('unit_file',
    help='unit data to summarize for as semicolon-delimited CSV'
)
parser.add_argument('out_file',
    help='path to output unit data with summarized interactions as semicolon-delimited CSV'
)
parser.add_argument('-i', '--unit-id-col', default='id',
    help='name of the unit ID attribute in the unit file'
)
parser.add_argument('-r', '--unit-region-col', default='region',
    help='name of the region ID attribute in the unit file'
)
parser.add_argument('-c', '--unit-core-col', default='is_core',
    help='name of the core indicator attribute in the unit file'
)
parser.add_argument('-l', '--distinguish-largest', action='store_true',
    help='distinguish the largest unit in region by ingoing interaction size'
)
parser.add_argument('-S', '--exclude-self-interactions', action='store_true',
    help='exclude self-interactions from summarization'
)


if __name__ == '__main__':
    args = parser.parse_args()
    unit_df = pd.read_csv(args.unit_file, sep=';')
    inter_df = pd.read_csv(args.inter_file, sep=';')
    unit_cols = [args.unit_id_col, args.unit_region_col]
    if args.unit_core_col:
        unit_cols.append(args.unit_core_col)
    if args.exclude_self_interactions:
        inter_df = inter_df[
            inter_df[args.from_id_col] != inter_df[args.to_id_col]
        ].copy()
    full_df = inter_df.merge(
        unit_df[unit_cols].rename(columns=dict(zip(unit_cols, [c + '_from' for c in unit_cols]))),
        left_on=args.from_id_col,
        right_on=args.unit_id_col + '_from',
        how='inner',
        suffixes=(False, False)
    ).merge(
        unit_df[unit_cols].rename(columns=dict(zip(unit_cols, [c + '_to' for c in unit_cols]))),
        left_on=args.to_id_col,
        right_on=args.unit_id_col + '_to',
        how='left',
        suffixes=(False, False)
    )
    reg_from_col = args.unit_region_col + '_from'
    reg_to_col = args.unit_region_col + '_to'
    full_df['_type'] = 'outof_region'
    is_within = full_df[reg_from_col] == full_df[reg_to_col]
    full_df.loc[is_within,'_type'] = 'within_region'
    if args.unit_core_col:
        full_df.loc[
            is_within & full_df[args.unit_core_col + '_to'],
            '_type'
        ] = 'to_core'
        full_df.loc[full_df['_type'] == 'within_region','_type'] = 'to_hinterland'
    if args.distinguish_largest:
        largest_units = full_df.groupby(
            [reg_to_col, args.to_id_col]
        )[args.strength_col].sum().reset_index(level=0).groupby(
            reg_to_col
        )[args.strength_col].idxmax().rename('largest_unit')
        with_largest = full_df.merge(
            largest_units, left_on=reg_from_col, right_index=True, how='left'
        )
        full_df.loc[
            with_largest[args.to_id_col] == with_largest['largest_unit'],
            '_type'
        ] = 'to_largest'
    inter_types = full_df['_type'].unique().tolist()
    out_inter_types = [args.strength_col + '_' + it for it in inter_types]
    type_df = full_df.groupby([args.from_id_col, '_type'])[args.strength_col].sum().reset_index(level=1).pivot(columns='_type', values=args.strength_col).fillna(0).rename(columns=dict(zip(inter_types, out_inter_types)))
    for col in type_df:
        if 'to_largest' in col:
            core_col = col.replace('to_largest', 'to_core')
            if core_col in type_df:
                type_df[core_col] += type_df[col]
    out_df = unit_df.set_index(args.unit_id_col).merge(type_df, left_index=True, right_index=True, how='left', suffixes=(False, False))
    out_df[args.strength_col + '_out_sum'] = inter_df.groupby(args.from_id_col)[args.strength_col].sum()
    out_df[args.strength_col + '_in_sum'] = inter_df.groupby(args.to_id_col)[args.strength_col].sum()
    out_inter_types += [
        args.strength_col + '_out_sum',
        args.strength_col + '_in_sum',
    ]
    for outcol in out_inter_types:
        modifcol = out_df[outcol].fillna(0)
        intcol = modifcol.astype(int)
        out_df[outcol] = intcol if (intcol == modifcol).all() else modifcol
    out_df.reset_index().to_csv(args.out_file, sep=';', index=False)
