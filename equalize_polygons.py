"""Merge and/or split polygons so that their surface area closely matches the target figure.

If a polygon is larger than target_area and subdivisions are available,
it is split into its subdivisions, which are reaggregated into compact shapes
to match target_area. If a polygon is smaller than target_area, it is
aggregated with its neighbors.
The criterion being minimized is ``abs(1 - area / target_area)``.

An interface to ``mobilib.area.equalize_polygons``.
"""

import geopandas as gpd

import mobilib.area
import mobilib.argparser
import mobilib.core

parser = mobilib.argparser.default(__doc__, areas=True)
parser.add_argument('-s', '--subdiv-file',
    help='areas to use as subdivisions of equalized areas as a GDAL-compatible polygon file'
)
parser.add_argument('-t', '--target-area', type=float,
    help='target area to approximate by the output polygons'
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
        unsafe_geom=args.unsafe_geom,
    )
    mobilib.core.write_gdf(gpd.GeoDataFrame(geometry=areas_eq), args.out_file)
