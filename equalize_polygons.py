import geopandas as gpd

import mobilib.area
import mobilib.argparser

parser = mobilib.argparser.default(__doc__, areas=True)
parser.add_argument('-s', '--subdiv-file',
    help='areas to use as subdivisions of equalized areas as a GDAL-compatible polygon file'
)
parser.add_argument('-p', '--subdiv-poly-id-col',
    help='column in the subdivisions file mapping to area IDs'
)
parser.add_argument('-u', '--unsafe-geom', action='store_true',
    help='the geometries of areas and subdivisions do not perfectly align, compensate for that'
)
parser.add_argument('out_file',
    help='path to output the GDAL-compatible polygon layer with areas equalized'
)

if __name__ == '__main__':
    args = parser.parse_args()
    areas = gpd.read_file(args.area_file).set_index(args.area_id_col)
    subdiv_geom = None
    subdiv_poly_ids = None
    if args.subdiv_file:
        subdivs = gpd.read_file(args.subdiv_file)
        subdiv_geom = subdivs.geometry
        if args.subdiv_poly_id_col:
            subdiv_poly_ids = subdivs[args.subdiv_poly_id_col]
    areas_eq, id_mapping = mobilib.area.equalize_polygons(
        areas.geometry,
        subdivisions=subdiv_geom,
        # subdiv_map=subdiv_poly_ids,
        unsafe_geom=args.unsafe_geom,
    )
    gpd.GeoDataFrame(geometry=areas_eq).to_file(args.out_file)
    # areas.geometry = areas.geometry.map(lambda x: x.representative_point())
    # # print(dir(areas.geometry))
    # areas['wkt'] = areas.geometry.map(lambda x: x.wkt)
    # areas['x'] = areas.geometry.x
    # areas['y'] = areas.geometry.y
    # areas.drop(['geometry'], axis=1).to_csv(args.out_file, sep=';', index=False)
