import geopandas as gpd

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
    areas = gpd.read_file(args.area_file)
    areas.geometry = areas.geometry.map(lambda x: x.representative_point())
    # print(dir(areas.geometry))
    areas['wkt'] = areas.geometry.map(lambda x: x.wkt)
    areas['x'] = areas.geometry.x
    areas['y'] = areas.geometry.y
    areas.drop(['geometry'], axis=1).to_csv(args.out_file, sep=';', index=False)
