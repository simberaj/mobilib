"""Transform a geospatial file to a GeoCSV (WKT)."""

import operator

import geopandas as gpd

import mobilib.argparser


parser = mobilib.argparser.default(__doc__)
parser.add_argument('in_file', help='GDAL-compatible file')
parser.add_argument('out_file', help='path to output CSV')

if __name__ == '__main__':
    args = parser.parse_args()
    in_gdf = gpd.read_file(args.in_file)
    if (in_gdf.geometry.geom_type == 'Point').all():
        in_gdf['X'] = in_gdf.geometry.x
        in_gdf['Y'] = in_gdf.geometry.y
    else:
        in_gdf['WKT'] = in_gdf.geometry.map(operator.attrgetter('wkt'))
    in_gdf.drop('geometry', axis=1).to_csv(args.out_file, sep=';', index=False)
