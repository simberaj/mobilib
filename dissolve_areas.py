'''Dissolve areas through a given mapping CSV.'''

import argparse

import pandas as pd
import geopandas as gpd


parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('areafile',
    help='GDAL-compatible polygon file with areas to dissolve'
)
parser.add_argument('mapfile',
    help='CSV file mapping polygon IDs to IDs of the dissolved objects'
)
parser.add_argument('outfile',
    help='path to output file with dissolved areas'
)
parser.add_argument('-a', '--area-id', default='id',
    help='name of the area ID attribute in the area file'
)
parser.add_argument('-s', '--source-id', default='id',
    help='name of the area ID attribute in the mapping file'
)
parser.add_argument('-t', '--target-id', default='region',
    help='name of the target aggregate ID attribute in the mapping file'
)

if __name__ == '__main__':
    args = parser.parse_args()
    mapfile = pd.read_csv(args.mapfile, sep=';')
    areafile = gpd.read_file(args.areafile)
    if areafile[args.area_id].dtype != mapfile[args.source_id].dtype:
        areafile[args.area_id] = areafile[args.area_id].astype(mapfile[args.source_id].dtype)
    joined = pd.merge(
        areafile, mapfile,
        how='left',
        left_on=args.area_id,
        right_on=args.source_id
    )
    dissolved = joined[['geometry', args.target_id]].dissolve(by=args.target_id).reset_index()
    dissolved.geometry = dissolved.geometry.apply(lambda poly: poly.simplify(.1, False))
    dissolved.to_file(args.outfile)