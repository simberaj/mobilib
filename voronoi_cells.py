"""Convert a point place file to a cell polygon file using Voronoi tesselation.

Optionally, constrains the polygons to a given extent.
"""

import mobilib.argparser
import mobilib.voronoi


parser = mobilib.argparser.default(__doc__, places=True, add_places_id=False)
parser.add_argument('out_file',
    help='path to output the cells as a GDAL-compatible or CSV WKT polygon file'
)
parser.add_argument('-e', '--extent',
    help='GDAL-compatible polygon file to which to clip the output'
)
parser.add_argument('-r', '--target-srid', type=int, default=3035,
    help='EPSG ID of the coordinate system to use on output'
)

if __name__ == '__main__':
    args = parser.parse_args()
    site_gdf = mobilib.core.read_places(args)
    site_gdf = site_gdf[~site_gdf.geometry.isnull()]
    if args.extent is not None:
        extent = mobilib.core.load_extent(args.extent, args.target_srid)
    else:
        extent = None
    if args.target_srid != args.source_srid:
        site_gdf = site_gdf.to_crs(mobilib.core.srid_to_crsdef(args.target_srid))
    site_gdf['cell'] = list(mobilib.voronoi.cells_shapely(site_gdf.geometry, extent))
    out_gdf = (
        site_gdf
        .rename(columns={'geometry' : 'point', 'cell' : 'geometry'})
        .drop(['point'], axis=1)
    )
    mobilib.core.write_gdf(out_gdf, args.out_file)
