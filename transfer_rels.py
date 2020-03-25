'''Apply an areal interpolation transfer table to a given field in an
interaction table, transferring from one set of identifiers to another.

Can possibly be used on multiple fields.

The input table should contain relations matching two identifiers (source and
target of the relation) and one or more

The transfer table should contain three fields - the first is the source
identifier matching the input file ID field, the second is the target identifier
that will be produced in the output and the third is the interpolation
(transfer) weight.

The self-interaction parameter value specifies the fraction of interactions
to be directly assigned to self-interactions instead of applying the relational
weighting.
'''

import argparse

import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('input_file',
    help='input data as semicolon-delimited CSV'
)
parser.add_argument('trans_table',
    help='transfer table as semicolon-delimited CSV'
)
parser.add_argument('output_file',
    help='path to output the interpolated CSV'
)
parser.add_argument('-f', '--from-id-field', default='from_id',
    help='name of the source area ID attribute in the input relations file'
)
parser.add_argument('-t', '--to-id-field', default='to_id',
    help='name of the target area ID attribute in the input relations file'
)
parser.add_argument('-a', '--abs-field', nargs='+', default=[],
    help='input file fields to aggregate by sum (absolute)'
)
parser.add_argument('-e', '--eta-val', type=float, default=0,
    help='the self-interaction parameter value'
)
parser.add_argument('-I', '--intraflow-eta', nargs='?', const='', default=None,
    help='calculate self-interaction parameter from intraflow fraction of given field'
)


def transfer(indf, transdf, from_id_col, to_id_col, abs_fields):
    trans_source, trans_target, trans_weight = transdf.columns[:3]
    transdf = transdf.set_index(trans_source)
    indf = indf.merge(
        transdf.rename(columns={
            trans_target: 'from_' + trans_target,
            trans_weight: 'from_weight',
        }),
        left_on=from_id_col, right_index=True,
        how='left', suffixes=(False, False)
    ).merge(
        transdf.rename(columns={
            trans_target: 'to_' + trans_target,
            trans_weight: 'to_weight'
        }).drop('eta', axis=1),
        left_on=to_id_col, right_index=True,
        how='left', suffixes=(False, False)
    )
    is_source_selfint = indf[from_id_col] == indf[to_id_col]
    is_target_selfint = indf['from_' + trans_target] == indf['to_' + trans_target]
    indf['_weighter'] = indf['from_weight'] * np.where(
        is_source_selfint,
        (is_target_selfint * indf['eta'] + indf['to_weight'] * (1 - indf['eta'])),
        indf['to_weight']
    )
    groupby_fields = ['from_' + trans_target, 'to_' + trans_target]
    all_agg_fields = abs_fields
    for val_field in all_agg_fields:
        indf[val_field] *= indf['_weighter']
    aggs = {field : 'sum' for field in all_agg_fields}
    return indf.groupby(groupby_fields).agg(aggs).reset_index()


def calculate_eta_intraflow(indf, transdf, from_id_col, to_id_col, val_col):
    intra = indf.loc[
        indf[from_id_col] == indf[to_id_col], [from_id_col, val_col]
    ].merge(
        indf.groupby(from_id_col)[val_col].sum(),
        left_on=from_id_col,
        right_index=True,
        how='outer',
        suffixes=('_intra', '_all'),
    ).set_index(from_id_col).fillna({
        val_col + '_intra': 0,
        val_col + '_all': 1,
    })
    intra['eta'] = intra[val_col + '_intra'] / intra[val_col + '_all']
    transdf = transdf.merge(
        intra['eta'],
        left_on=transdf.columns[0],
        right_index=True,
    )
    return transdf


if __name__ == '__main__':
    args = parser.parse_args()
    indf = pd.read_csv(args.input_file, sep=';')
    transdf = pd.read_csv(args.trans_table, sep=';')
    if 'eta' not in transdf:
        if args.intraflow_eta is not None:
            transdf = calculate_eta_intraflow(
                indf, transdf,
                args.from_id_field, args.to_id_field,
                (args.intraflow_eta if args.intraflow_eta else args.abs_field[0])
            )
        else:
            transdf['eta'] = args.eta_val
    outdf = transfer(
        indf, transdf,
        args.from_id_field, args.to_id_field,
        args.abs_field
    )
    outdf.to_csv(args.output_file, sep=';', index=False)
