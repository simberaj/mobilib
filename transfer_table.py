
'''Create a transfer table to perform areal interpolation between two sets of units.

A transfer table specifies what share of a given source area values is to
be transferred to a given target area. This can be computed using their overlaps
which can be weighted according to a given weighting layer.

Optionally, a transfer table can also specify the self-interaction parameter
for the given 
'''

import argparse
import pickle

import numpy
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
parser.add_argument('weighting_file',
    help='weighting value layer as a GDAL-compatible polygon file'
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
parser.add_argument('-w', '--weight-field', default='weight',
    help='field in weighting layer containing the (absolute) weights'
)
parser.add_argument('-m', '--eta-model',
    help='a model to estimate the self-interaction parameter'
)

if __name__ == '__main__':
    args = parser.parse_args()
    source_gdf = gpd.read_file(args.source_file)
    weight_gdf = gpd.read_file(args.weighting_file).rename_axis('wt_id').reset_index()
    weight_gdf['wt_geom'] = weight_gdf['geometry']
    parts_gdf = gpd.sjoin(source_gdf, weight_gdf, op='intersects', how='left')
    parts_gdf['part'] = parts_gdf.apply(intersection_fx('geometry', 'wt_geom'), axis=1)
    parts_gdf['part_area'] = parts_gdf['part'].area
    parts_gdf = parts_gdf.merge(
        parts_gdf.groupby('wt_id')['part_area'].sum().reset_index().rename(
            columns={'part_area' : 'part_area_sum'}
        ),
        on='wt_id',
        how='left'
    ) 
    parts_gdf['part_weight'] = (
        parts_gdf[args.weight_field].fillna(1)
        * parts_gdf['part_area']
        / parts_gdf['part_area_sum']
    )
    parts_gdf['is_orphan'] = parts_gdf[args.weight_field].isna()
    parts_gdf.loc[parts_gdf['is_orphan'],'wt_id'] = (
        parts_gdf.index[parts_gdf['is_orphan']]
        + parts_gdf['wt_id'].max()
    )
    parts_gdf = parts_gdf.merge(
        parts_gdf.groupby(args.source_id_field).agg({
            'part_weight' : 'sum',
            'geometry' : 'count',
        }).rename(columns={
            'part_weight' : 'part_weight_source_sum',
            'geometry' : 'part_source_count',
        }),
        on=args.source_id_field,
    )
    parts_gdf['source_weight'] = parts_gdf['part_weight'].fillna(1) / numpy.where(
        parts_gdf['part_weight_source_sum'] == 0,
        parts_gdf['part_source_count'],
        parts_gdf['part_weight_source_sum']
    )
    disag_sources = pd.concat([
        weight_gdf.loc[:,['geometry', 'wt_id']],
        parts_gdf.loc[parts_gdf['is_orphan'],['geometry', 'wt_id']],
    ]).rename(columns={'geometry' : 'src_geom'})
    disag_sources_gdf = gpd.GeoDataFrame(
        disag_sources,
        geometry=disag_sources['src_geom'],
        crs=source_gdf.crs
    )
    target_gdf = gpd.read_file(args.target_file)
    target_gdf['tgt_geom'] = target_gdf['geometry']
    agg_gdf = gpd.sjoin(target_gdf, disag_sources_gdf, op='intersects')
    agg_gdf['part_area'] = agg_gdf.apply(intersection_area_fx('src_geom', 'tgt_geom'), axis=1)
    agg_gdf = agg_gdf.merge(
        agg_gdf.groupby('wt_id')['part_area'].sum().reset_index().rename(columns={
            'part_area' : 'part_area_sum'
        }),
        on='wt_id'
    )
    agg_gdf['target_weight'] = agg_gdf['part_area'] / agg_gdf['part_area_sum']
    fin_gdf = pd.merge(
        parts_gdf[['wt_id', args.source_id_field, 'source_weight']],
        agg_gdf[['wt_id', args.target_id_field, 'target_weight']],
        on='wt_id',
        suffixes=('_from', '_to'),
    )
    fin_gdf['weight'] = fin_gdf['source_weight'] * fin_gdf['target_weight']
    trans_table_pre = fin_gdf.groupby(
        [args.source_id_field, args.target_id_field]
    )['weight'].sum().reset_index()
    trans_table = pd.merge(
        trans_table_pre,
        trans_table_pre.groupby(args.source_id_field)['weight'].sum().reset_index().rename(columns={'weight' : 'weight_sum'}),
        on=args.source_id_field
    )
    trans_table['weight'] /= trans_table['weight_sum']
    trans_table = trans_table.drop('weight_sum', axis=1)
    print(trans_table)
    trans_table.to_csv(args.out_table, sep=';', index=False)
