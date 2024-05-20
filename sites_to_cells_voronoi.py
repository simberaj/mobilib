
'''Converts a site CSV to cell polygon file using Voronoi tesselation,
optionally constraining to a given extent.'''

import argparse

import pandas as pd

import mobilib
import mobilib.voronoi


parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('sitefile',
    help='input site locations as semicolon-delimited CSV'
)
parser.add_argument('outcellfile',
    help='path to output the cells as a GDAL-compatible polygon file'
)
parser.add_argument('-e', '--extent',
    help='GDAL-compatible polygon file to which to clip the output'
)
parser.add_argument('-x', '--xcol', default='x',
    help='name of the X-coordinate column in site file'
)
parser.add_argument('-y', '--ycol', default='y',
    help='name of the Y-coordinate column in site file'
)
parser.add_argument('-s', '--source-srid', type=int, default=4326,
    help='EPSG ID of the coordinate system of the site coordinates'
)
parser.add_argument('-r', '--target-srid', type=int, default=3035,
    help='EPSG ID of the coordinate system to use on output'
)

if __name__ == '__main__':
    args = parser.parse_args()
    site_gdf = mobilib.point_gdf(
        pd.read_csv(args.sitefile, sep=';'),
        xcol=args.xcol,
        ycol=args.ycol,
        srid=args.source_srid,
    )
    # print(dir(site_gdf.geometry))
    site_gdf = site_gdf[~site_gdf.geometry.isnull()]
    extent = mobilib.load_extent(args.extent, args.target_srid)
    if args.target_srid != args.source_srid:
        site_gdf = site_gdf.to_crs(mobilib.srid_to_crsdef(args.target_srid))
    site_gdf['cell'] = list(mobilib.voronoi.cells_shapely(site_gdf.geometry, extent))
    site_gdf.rename(
        columns={'geometry' : 'point', 'cell' : 'geometry'}
    ).drop(['point'], axis=1).to_file(args.outcellfile)
