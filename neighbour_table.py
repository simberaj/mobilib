
'''Generates location relation tables from an anchor point table.'''

import argparse

import pandas as pd
import geopandas as gpd

import mobilib.neigh



parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('infile',
    help='input polygon table as a semicolon-delimited CSV with WKT geometries'
)
parser.add_argument('outfile',
    help='path to output neighbourhood table'
)
# parser.add_argument('-g', '--geom',
    # default='geometry', help='geometry column name', metavar='COLNAME'
# )
parser.add_argument('-i', '--id',
    default=None, help='area id column', metavar='COLNAME'
)


if __name__ == '__main__':
    args = parser.parse_args()
    data = gpd.read_file(args.infile, encoding='utf8')
    geoms = data.geometry.tolist()
    ids = None if args.id is None else data[args.id]
    neighs = list(mobilib.neigh.neighbours(geoms, ids))
    df = pd.DataFrame.from_records(neighs, columns=['from_id', 'to_id'])
    df.to_csv(args.outfile, sep=';', index=False)
