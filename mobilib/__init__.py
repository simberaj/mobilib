
'''Generic, all-purpose functions.'''

import functools
import argparse

import numpy
import pyproj
import shapely.geometry
import shapely.ops
import pandas as pd
import geopandas as gpd

check_geom_cols = ['geometry', 'wkt']

def srid_to_crsdef(srid):
    return {'init' : 'epsg:' + str(srid)}


def proj(crsdef):
    return pyproj.Proj(**crsdef)


def srid_proj(srid):
    return pyproj.Proj(srid)


def transformation(from_proj, to_proj):
    projtrans = functools.partial(pyproj.transform, from_proj, to_proj)
    shtrans = shapely.ops.transform
    return lambda geom: shtrans(projtrans, geom)


def point_gdf(df, xcol='X', ycol='Y', srid=4326, drop_locs=False):
    return gpd.GeoDataFrame(
        (df.drop([xcol, ycol], axis=1) if drop_locs else df),
        crs=srid_to_crsdef(srid),
        geometry=[
            shapely.geometry.Point(xy)
                if not numpy.isnan(xy).sum() else shapely.geometry.Point()
            for xy in zip(df[xcol], df[ycol])
        ]
    )

def wkt_gdf(df, wkt_col=check_geom_cols[0], srid=4326, drop_locs=True):
    return gpd.GeoDataFrame(
        (df.drop(wkt_col, axis=1) if drop_locs else df),
        crs=srid_to_crsdef(srid),
        geometry=df[wkt_col].map(shapely.wkt.loads),
    )

def load_extent(path=None, target_srid=None):
    if path:
        extent_df = gpd.read_file(path)
        poly = extent_df.loc[extent_df.index[0],'geometry']
        if target_srid:
            target_proj = srid_proj(target_srid)
            source_proj = proj(extent_df.crs)
            if source_proj != target_proj:
                poly = transformation(source_proj, target_proj)(poly)
        return poly
    else:
        return None


def read_gdf(path, xcol='X', ycol='Y', srid=4326):
    if path.endswith('.csv'):
        return read_csv_gdf(path, xcol, ycol, srid)
    else:
        return gpd.read_file(path)


def write_gdf(gdf, path):
    if path.endswith('.csv'):
        gdf.to_csv(path, sep=';', index=False)
    else:
        gdf.to_file(path)


def read_csv_gdf(path, xcol='X', ycol='Y', srid=4326):
    df = pd.read_csv(path, sep=';')
    cols = [col.lower() for col in df.columns.tolist()]
    for col in check_geom_cols:
        if col in cols:
            return wkt_gdf(df, df.columns[cols.index(col)], srid)
    else:
        return point_gdf(df, xcol, ycol, srid)


def read_places(args):
    return read_gdf(args.place_file, args.x_col, args.y_col, args.srid)


def read_nonspatial(path, **kwargs):
    if path.endswith('.csv'):
        return pd.read_csv(path, sep=';', **kwargs)
    else:
        return pd.DataFrame(gpd.read_file(path).drop('geometry', axis=1))
