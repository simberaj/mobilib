import geopandas as gpd

import mobilib
import mobilib.area
import mobilib.argparser

parser = mobilib.argparser.default(__doc__)
parser.add_argument('area_file',
    help='interacting areas layer as a GDAL-compatible polygon file'
)
parser.add_argument('out_file',
    help='path to output the CSV with area attributes and representative point coordinates (in x, y and point columns)'
)

if __name__ == '__main__':
    args = parser.parse_args()
    mobilib.point_gdf_to_wkt(
        mobilib.area.representative_points(gpd.read_file(args.area_file))
    ).to_csv(args.out_file, sep=';', index=False)
