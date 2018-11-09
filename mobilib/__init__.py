
'''Generic, all-purpose functions.'''

import functools

import numpy
import pyproj
import shapely.geometry
import shapely.ops
import geopandas as gpd


def srid_to_crsdef(srid):
    return {'init' : 'epsg:' + str(srid)}


def proj(crsdef):
    return pyproj.Proj(**crsdef)


def srid_proj(srid):
    return pyproj.Proj(**srid_to_crsdef(srid))


def transformation(from_proj, to_proj):
    projtrans = functools.partial(pyproj.transform, from_proj, to_proj)
    shtrans = shapely.ops.transform
    return lambda geom: shtrans(projtrans, geom)


def point_gdf(df, xcol='x', ycol='y', srid=4326, drop_locs=False):
    return gpd.GeoDataFrame(
        (df.drop([xcol, ycol], axis=1) if drop_locs else df),
        crs=srid_to_crsdef(srid),
        geometry=[
            shapely.geometry.Point(xy)
                if not numpy.isnan(xy).sum() else shapely.geometry.Point()
            for xy in zip(df[xcol], df[ycol])
        ]
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