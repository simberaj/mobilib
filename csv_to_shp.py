"""Transform a CSV (GeoCSV) to a shapefile or other GDAL-compatible file."""

import mobilib.argparser
import mobilib.core

parser = mobilib.argparser.default(__doc__)
parser.add_argument(
    'in_file',
    help='semicolon-delimited spatial CSV'
)
parser.add_argument(
    '-x', '--x-col', default='X',
    help='name of the x-coordinate attribute in the input file (only use if no WKT column present)'
)
parser.add_argument(
    '-y', '--y-col', default='Y',
    help='name of the y-coordinate attribute in the input file (only use if no WKT column present)'
)
parser.add_argument(
    '-c', '--srid', default=4326,
    help='EPSG SRID of the input file coordinates'
)
parser.add_argument(
    'out_file',
    help='path to output shapefile'
)

if __name__ == '__main__':
    args = parser.parse_args()
    args.place_file = args.in_file
    mobilib.core.write_gdf(mobilib.core.read_places(args), args.out_file)
