"""Create a transfer table to perform areal interpolation by areal weighting.

A transfer table specifies what share of a given source area values is to
be transferred to a given target area. This can be computed using their
area overlaps, which is a simple and very inaccurate method but has no
dependencies on external data, making it always available.
"""

import geopandas as gpd

import mobilib.argparser


parser = mobilib.argparser.default(__doc__)
parser.add_argument(
    'source_file',
    help='source value layer as a GDAL-compatible polygon file'
)
parser.add_argument(
    'target_file',
    help='target area layer as a GDAL-compatible polygon file'
)
parser.add_argument(
    'out_table',
    help='path to output the transfer table as a semicolon-delimited CSV'
)
parser.add_argument(
    '-s', '--source-id-field', default='id',
    help='ID field of the source layer (will be used in the transfer table)'
)
parser.add_argument(
    '-t', '--target-id-field', default='id',
    help='ID field of the target layer (will be used in the transfer table)'
)

if __name__ == '__main__':
    args = parser.parse_args()
    source_geoms = gpd.read_file(args.source_file).set_index(args.source_id_field)['geometry']
    target_geoms = gpd.read_file(args.target_file).set_index(args.target_id_field)['geometry']
    inters_gdf = gpd.overlay(
        gpd.GeoDataFrame({'geometry': source_geoms, args.source_id_field: source_geoms.index}),
        gpd.GeoDataFrame({'geometry': target_geoms, args.target_id_field: target_geoms.index}),
        how='intersection',
    )
    inters_gdf['_area'] = inters_gdf['geometry'].area
    for_coefs_df = inters_gdf.drop('geometry', axis=1).join(
        inters_gdf.groupby(args.source_id_field)['_area'].sum().rename('_source_area'),
        on=args.source_id_field,
    )
    for_coefs_df['weight'] = for_coefs_df.eval('_area / _source_area')
    for_coefs_df.drop(
        ['_area', '_source_area'], axis=1
    )[[args.source_id_field, args.target_id_field, 'weight']].sort_values(
        [args.source_id_field, args.target_id_field]
    ).to_csv(
        args.out_table, sep=';', index=False
    )
