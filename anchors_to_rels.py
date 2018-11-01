
'''Generates location relation tables from an anchor point table.'''

import argparse

import numpy
import pandas as pd

import mobilib.relations as rels



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
    default=rels.DEFAULT_HOME_CODE,
    metavar='CODE',
    help='home anchor point type code'
)
parser.add_argument('-W', '--work',
    default=rels.DEFAULT_WORK_CODE,
    metavar='CODE',
    help='work anchor point type code'
)
parser.add_argument('-M', '--multi',
    default=rels.DEFAULT_MULTIFX_CODE,
    metavar='CODE',
    help='multifunctional anchor point type code'
)

if __name__ == '__main__':
    args = parser.parse_args()
    home_codes = rels.build_codes(args.home, args.multi)
    work_codes = rels.build_codes(args.work, args.multi)
    geners = [
        rels.GeneralRelationGenerator(),
        rels.HomeBaseRelationGenerator(home_codes),
        rels.HomeWorkRelationGenerator(home_codes, work_codes),
    ]
    df = pd.read_csv(args.infile, sep=';')
    sites = numpy.sort(df[args.site_id].unique())
    n_sites = len(sites)
    for gener in geners:
        matrix = numpy.zeros((n_sites, n_sites))
        for uid, subdf in df.groupby(args.user_id):
            site_indexes = numpy.searchsorted(sites, subdf[args.site_id].values)
            rels = gener.relate(
                subdf[args.importance].values,
                subdf[args.type].values
            )
            for i in site_indexes:
                matrix[i,site_indexes] += rels[i]
        rowis, colis = numpy.nonzero(matrix)
        outfname = args.outfile + '_' + gener.name + '.csv'
        pd.DataFrame({
            'from_' + args.site_id : sites[rowis],
            'to_' + args.site_id : sites[colis],
            args.importance : matrix[rowis,colis],
        }).to_csv(outfname, index=False, sep=';')