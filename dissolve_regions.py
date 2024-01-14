"""Dissolve units to regions through a given mapping CSV.

Handles multimember cores and naming by largest unit.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely.geometry

import mobilib.argparser


def merge_areas(unit_df, area_gdf, unit_id, area_id):
    if area_gdf[area_id].dtype != unit_df[unit_id].dtype:
        area_gdf[area_id] = area_gdf[area_id].astype(unit_df[unit_id].dtype)
    unit_cols = unit_df.columns.tolist()
    unit_df = unit_df.merge(
        area_gdf.drop([
            col for col in area_gdf.columns
            if col in unit_cols and col != area_id
        ], axis=1),
        how='outer',
        left_on=unit_id,
        right_on=area_id,
    )
    unit_df[unit_id].fillna(unit_df[area_id], inplace=True)
    return unit_df


def dissolve_areas(unit_df, reg_col):
    return gpd.GeoDataFrame(unit_df, geometry='geometry').dropna(
        subset=['geometry']
    )[['geometry', reg_col]].dissolve(by=reg_col)['geometry'].apply(fix_holes)


def fix_holes(poly):
    if hasattr(poly, 'geoms'):
        return shapely.geometry.MultiPolygon([fix_holes(geom) for geom in poly.geoms])
    else:
        holes = [
            hole for hole in poly.interiors
            if shapely.geometry.Polygon(hole).area >= .1
        ]
        if len(holes) < len(poly.interiors):
            return shapely.geometry.Polygon(poly.exterior, holes)
        else:
            return poly


def aggregate_attrs(unit_df, agg_cols, reg_col, agg_func, prefix='', count_units=True):
    agg_dict = {
        prefix + col: pd.NamedAgg(column=col, aggfunc=agg_func)
        for col in agg_cols
    }
    if count_units:
        agg_dict[prefix + 'unit_count'] = pd.NamedAgg(column=reg_col, aggfunc='count')
    return unit_df.groupby(reg_col).aggregate(**agg_dict)


def to_wkt(geom):
    return geom.wkt


parser = mobilib.argparser.default(__doc__)
parser.add_argument(
    'unit_file',
    help='semicolon-delimited CSV with unit data and mapping to regions'
)
parser.add_argument(
    'out_file',
    help='path to output file with dissolved regions'
)
parser.add_argument(
    '-a', '--area-file',
    help='CSV file mapping polygon IDs to IDs of the dissolved objects'
)
parser.add_argument(
    '-u', '--unit-id-col', default='id',
    help='name of the unit ID attribute in the unit file'
)
parser.add_argument(
    '-r', '--unit-region-col', default='region',
    help='name of the region ID attribute in the unit file'
)
parser.add_argument(
    '-c', '--unit-core-col', default='is_core',
    help='name of the core indicator attribute in the unit file'
)
parser.add_argument(
    '-U', '--area-unit-id-col',
    help='name of the area ID attribute in the area file'
         ' (default: same as --unit-id-col)'
)
parser.add_argument(
    '-s', '--sum-cols', nargs='+',
    help='unit columns to aggregate by sum to regions'
)
parser.add_argument(
    '-l', '--distinguish-largest-by-col', default='mass',
    help='distinguish largest unit of region by this attribute'
)
parser.add_argument(
    '-H', '--distinguish-hinterland', action='store_true',
    help='unit columns to aggregate by sum to regions'
)

if __name__ == '__main__':
    args = parser.parse_args()
    if not args.sum_cols:
        raise ValueError('need at least one column to summarize')
    unit_df = pd.read_csv(args.unit_file, sep=';')
    is_core_def = args.unit_core_col and args.unit_core_col in unit_df
    is_largdisting_def = args.distinguish_largest_by_col and args.distinguish_largest_by_col in unit_df
    if args.area_file:
        area_id_col = (
            args.area_unit_id_col if args.area_unit_id_col is not None
            else args.unit_id_col
        )
        unit_df = merge_areas(
            unit_df, gpd.read_file(args.area_file),
            args.unit_id_col, area_id_col,
        )
    unit_df = unit_df[unit_df[args.unit_region_col].notna()]
    if args.area_file:
        if is_core_def:
            unit_df[args.unit_core_col].fillna(True, inplace=True)
        if is_largdisting_def:
            unit_df[args.distinguish_largest_by_col].fillna(0, inplace=True)
        geometry = dissolve_areas(unit_df, args.unit_region_col)
    else:
        geometry = None
    sum_df = aggregate_attrs(unit_df, args.sum_cols, args.unit_region_col, 'sum')
    if is_core_def:
        sum_df = sum_df.merge(aggregate_attrs(
            unit_df[unit_df[args.unit_core_col]],
            args.sum_cols, args.unit_region_col, 'sum', prefix='core_'
        ), how='left', left_index=True, right_index=True)
        if args.distinguish_hinterland:
            hint_df = aggregate_attrs(
                unit_df[~unit_df[args.unit_core_col]],
                args.sum_cols, args.unit_region_col, 'sum', prefix='hinterland_'
            )
            sum_df = sum_df.merge(hint_df, how='left', left_index=True, right_index=True)
            for hint_col in hint_df.columns:
                sum_df[hint_col].fillna(0, inplace=True)
    if is_largdisting_def:
        max_ids = (
            unit_df
            .set_index(args.unit_id_col)
            .groupby(args.unit_region_col)
            [args.distinguish_largest_by_col].idxmax()
            .values
        )
        sum_df = sum_df.merge(aggregate_attrs(
            unit_df[unit_df[args.unit_id_col].isin(max_ids)],
            args.sum_cols, args.unit_region_col, 'sum', prefix='largest_', count_units=False
        ), how='left', left_index=True, right_index=True)
    sum_df.reset_index(inplace=True)
    if geometry is not None:
        sum_df = sum_df.merge(
            geometry.apply(to_wkt).rename("geometry"),
            left_on=args.unit_region_col,
            right_index=True,
            how='left'
        )
    sum_df.to_csv(args.out_file, sep=';', index=False)
