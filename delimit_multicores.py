'''Delimit multiple cores of regions based on interaction intensity.

Aggregate spatial units into groups that are joined by interactions over
specified intensity, selecting the largest member's identifier to represent
the group. A flow in one direction is sufficient; perform bidirectional
aggregation beforehand if you want decisions on bidirectional flows.
'''

import sys
import difflib
import logging
from typing import Any, List, Tuple, Set, Iterable

import pandas as pd

import mobilib
import mobilib.argparser


def to_joining_relations(df: pd.DataFrame,
                         from_col: str,
                         to_col: str,
                         ) -> Set[Tuple[Any, Any]]:
    return set(
        tuple(sorted(rec))
        for rec in df[[from_col, to_col]].itertuples(index=False, name=None)
    )


def create_groups(relations: Iterable[Tuple[Any, Any]]) -> List[Set[Any]]:
    unit_register = {}
    groups = []
    for one, two in relations:
        if one in unit_register:
            if two in unit_register:
                if unit_register[one] != unit_register[two]:
                    rem_group = unit_register[two]
                    # print(unit_register[one], '+=', unit_register[two])
                    unit_register[one] |= rem_group
                    for other in rem_group:
                        unit_register[other] = unit_register[one]
                    groups.remove(rem_group)
            else:
                # print(unit_register[one], '++', two)
                unit_register[one].add(two)
                unit_register[two] = unit_register[one]
        elif two in unit_register:
            # print(unit_register[two], '++', one)
            unit_register[two].add(one)
            unit_register[one] = unit_register[two]
        else:
            new_group = {one, two}
            unit_register[one] = new_group
            unit_register[two] = new_group
            # print('new', new_group)
            groups.append(new_group)
    return groups


def groups_to_headed_series(groups: Iterable[Set[Any]],
                            importances: pd.Series,
                            ) -> pd.Series:
    heads = {}
    for group in groups:
        head = importances.reindex(list(group)).idxmax()
        for item in group:
            heads[item] = head
    return pd.Series(heads)


def common_colname_part(one: str, two: str) -> str:
    matcher = difflib.SequenceMatcher(a=one, b=two)
    match = matcher.find_longest_match(0, len(one), 0, len(two))
    return one[match.a:(match.a+match.size)]


parser = mobilib.argparser.default(__doc__, interactions=True)
parser.add_argument('-S', '--min-strength',
    help='minimum interaction strength threshold to create multicores'
)
parser.add_argument('-I', '--inter-importance-col',
    help='interaction attribute to determine largest multicore member by largest inflow (default: equal to --strength-col)'
)
parser.add_argument('-u', '--unit-file',
    help='interacting unit data as a semicolon-delimited CSV'
)
parser.add_argument('-i', '--unit-id-col', default='id',
    help='name of the unit ID attribute (in the unit file and in the output)'
)
parser.add_argument('-C', '--preset-col',
    help='name of additional presets in the unit file to use as multicore joiners'
)
parser.add_argument('-U', '--unit-importance-col',
    help='unit attribute to determine largest multicore member by value'
)
parser.add_argument('out_file',
    help='path to output the multi-core assignments as a semicolon-delimited CSV'
)


if __name__ == '__main__':
    args = parser.parse_args()
    inter_df = pd.read_csv(args.inter_file, sep=';')
    multicore_df = inter_df.query('{0.strength_col} >= {0.min_strength}'.format(args))
    joining_relations = to_joining_relations(
        multicore_df, args.from_id_col, args.to_id_col
    )
    unit_df = None
    importances = None
    if args.unit_file:
        unit_df = pd.read_csv(args.unit_file, sep=';')
        if args.preset_col:
            joining_relations |= to_joining_relations(unit_df, args.unit_id_col, args.preset_col)
        if args.unit_importance_col:
            importances = unit_df.set_index(args.unit_id_col)[args.unit_importance_col]
    if importances is None:
        imp_col = args.inter_importance_col if args.inter_importance_col else args.strength_col
        importances = inter_df[
            inter_df[args.to_id_col].notna()
        ].groupby(args.to_id_col)[imp_col].sum()
    groups = create_groups(joining_relations)
    group_series = groups_to_headed_series(groups, importances).rename('multicore')
    if unit_df is None:
        all_unit_ids = inter_df[args.from_id_col].unique().tolist()
        if args.unit_id_col is None:
            id_col = common_colname_part(args.from_id_col, args.to_id_col)
            if id_col == '': id_col = 'id'
        else:
            id_col = args.unit_id_col
        out_df = group_series.reindex(all_unit_ids).rename_axis(id_col).reset_index()
    else:
        out_df = unit_df.merge(
            group_series,
            left_on=args.unit_id_col,
            right_index=True,
            how='left'
        )
    out_df.to_csv(args.out_file, sep=';', index=False)
