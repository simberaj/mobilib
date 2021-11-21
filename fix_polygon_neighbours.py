'''Fix geometries of neighboring polygons so that they align nicely.'''

import geopandas as gpd

import mobilib.argparser
import mobilib.neigh


parser = mobilib.argparser.default(__doc__)
parser.add_argument(
    'areafile',
    help='GDAL-compatible polygon file with areas to fix neighbourhood in'
)
parser.add_argument(
    'outfile',
    help='path to output a GDAL-compatible file with fixed areas'
)
parser.add_argument(
    '-t', '--tolerance', type=float, default=.01,
    help='tolerance to which to snap vertices of neighbours'
)

if __name__ == '__main__':
    args = parser.parse_args()
    areafile = gpd.read_file(args.areafile)
    areafile.geometry = mobilib.neigh.fix_polygons(areafile.geometry)
    areafile.to_file(args.outfile)
