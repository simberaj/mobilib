"""Apply an areal interpolation transfer table to simple data.

This allows extensive measures to be recomputed from one set of identifiers
to another. Can possibly be used on multiple fields.

The transfer table should contain three fields - the first is the source
identifier matching the input file ID field, the second is the target identifier
that will be produced in the output and the third is the interpolation
(transfer) weight.
"""

import argparse

import pandas as pd

import mobilib.argparser
import mobilib.core

parser = mobilib.argparser.default(__doc__)
parser.add_argument(
    'input_file',
    help='input data as semicolon-delimited CSV'
)
parser.add_argument(
    'trans_table',
    help='transfer table as semicolon-delimited CSV'
)
parser.add_argument(
    'output_file',
    help='path to output the interpolated CSV'
)
parser.add_argument(
    '-i', '--id-field', default='id',
    help='input file id column(s)'
)
parser.add_argument(
    '-g', '--grp-field', nargs='+', default=[],
    help='input file fields to group by in addition to ID field'
)
parser.add_argument(
    '-a', '--abs-field', nargs='+', default=[],
    help='input file fields to aggregate by sum (absolute)'
)

if __name__ == '__main__':
    args = parser.parse_args()
    indf = mobilib.core.read_nonspatial(args.input_file)
    transdf = pd.read_csv(args.trans_table, sep=';')
    trans_source, trans_target, trans_weight = transdf.columns[:3]
    indf = indf.merge(
        transdf,
        left_on=args.id_field,
        right_on=trans_source,
        how='left',
        suffixes=(None, None)
    )
    groupby_fields = [trans_target] + args.grp_field
    all_agg_fields = args.abs_field
    for val_field in all_agg_fields:
        indf[val_field] *= indf[trans_weight]
    aggs = {field : 'sum' for field in all_agg_fields}
    outdf = indf.groupby(groupby_fields).agg(aggs).reset_index()
    outdf.to_csv(args.output_file, sep=';', index=False)
