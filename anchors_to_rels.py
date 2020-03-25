
'''Generates location relation tables from an anchor point table.'''

import os
import argparse

import numpy
import pandas as pd

import mobilib.relations



parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('infile',
    help='input anchor point table as semicolon-delimited CSV'
)
parser.add_argument('outfile',
    help='path to output relation tables (will be suffixed by generator name\
    and .csv)'
)
parser.add_argument('-s', '--site-id',
    default='site_id', help='site or zone id column name', metavar='COLNAME'
)
parser.add_argument('-u', '--user-id',
    default='user_id', help='user (person) id column', metavar='COLNAME'
)
parser.add_argument('-i', '--importance',
    default='strength', help='anchor point importance column', metavar='COLNAME'
)
parser.add_argument('-t', '--type',
    default='type', help='anchor point type column', metavar='COLNAME'
)
parser.add_argument('-H', '--home',
    default=mobilib.relations.DEFAULT_HOME_CODE,
    metavar='CODE',
    help='home anchor point type code'
)
parser.add_argument('-W', '--work',
    default=mobilib.relations.DEFAULT_WORK_CODE,
    metavar='CODE',
    help='work anchor point type code'
)
parser.add_argument('-M', '--multi',
    default=mobilib.relations.DEFAULT_MULTIFX_CODE,
    metavar='CODE',
    help='multifunctional anchor point type code'
)
parser.add_argument('-x', '--exclude-selfinter', action='store_true',
    help='do not generate self-interactions'
)
parser.add_argument('-g', '--generator',
    help='single generator to use'
)


def generate_all(df, generators, site_id, user_id, importance, type, **kwargs):
    sites = numpy.sort(df[site_id].unique())
    n_sites = len(sites)
    series = []
    for gener in generators:
        print(gener.name)
        matrix = numpy.zeros((n_sites, n_sites))
        for uid, subdf in df.groupby(user_id):
            ids = subdf[site_id].values
            rels = gener.relate(
                subdf[importance].values,
                subdf[type].values,
                ids=ids,
            )
            site_indexes = numpy.searchsorted(sites, ids)
            if rels is not None:
                for src_i, tgt_i in enumerate(site_indexes):
                    matrix[tgt_i,site_indexes] += rels[src_i]
        rowis, colis = numpy.nonzero(matrix)
        series.append(pd.Series(
            matrix[rowis,colis],
            index=[sites[rowis],sites[colis]],
            name=gener.name,
        ))
    out_df = series[0]
    for ser in series[1:]:
        out_df = pd.merge(out_df, ser, left_index=True, right_index=True, how='outer')
    out_df.index.names = ['from_' + site_id, 'to_' + site_id]
    return out_df.fillna(0).reset_index()


if __name__ == '__main__':
    args = parser.parse_args()
    generators = mobilib.relations.default_generators(
        args.home, args.work, args.multi, not args.exclude_selfinter
    )
    if args.generator:
        generators = [g for g in generators if g.name == args.generator]
    df = pd.read_csv(args.infile, sep=';')
    rels_df = generate_all(df, generators, **vars(args))
    rels_df.to_csv(args.outfile, index=False, sep=';')
