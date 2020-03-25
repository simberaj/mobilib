
'''Create a transfer table to perform areal interpolation between two sets of units by areal weighting.

A transfer table specifies what share of a given source area values is to
be transferred to a given target area. This can be computed using their 
area overlaps.
'''

import argparse

import pandas as pd
import geopandas as gpd

import mobilib


def intersection_area_fx(gcol1, gcol2):
    def intersector(row):
        if row[gcol2] is numpy.nan:
            return row[gcol1].area
        else:
            return row[gcol1].intersection(row[gcol2]).area
    return intersector

def intersection_fx(gcol1, gcol2):
    def intersector(row):
        if row[gcol2] is numpy.nan:
            return np.nan
        else:
            return row[gcol1].intersection(row[gcol2])
    return intersector


parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('source_file',
    help='source value layer as a GDAL-compatible polygon file'
)
parser.add_argument('target_file',
    help='target area layer as a GDAL-compatible polygon file'
)
parser.add_argument('out_table',
    help='path to output the transfer table as a semicolon-delimited CSV'
)
parser.add_argument('-s', '--source-id-field', default='id',
    help='ID field of the source layer (will be used in the transfer table)'
)
parser.add_argument('-t', '--target-id-field', default='id',
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
    for_coefs_df = pd.merge(
        inters_gdf.drop('geometry', axis=1),
        inters_gdf.groupby(args.source_id_field)['_area'].sum().rename('_source_area'),
        left_on=args.source_id_field,
        right_index=True,
    )
    for_coefs_df['weight'] = for_coefs_df.eval('_area / _source_area')
    for_coefs_df.drop(
        ['_area', '_source_area'], axis=1
    )[[args.source_id_field, args.target_id_field, 'weight']].sort_values(
        [args.source_id_field, args.target_id_field]
    ).to_csv(
        args.out_table, sep=';', index=False
    )
