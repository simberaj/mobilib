"""Functions for library-wide use: common algorithms, spatial I/O, etc."""

import os
import warnings
import functools
from typing import Any, List, Dict, Callable, Union, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely.geometry
import shapely.geometry.base
import shapely.ops
import pyproj


CHECK_GEOM_COLS: List[str] = ['geometry', 'wkt']


def ipf(values: np.ndarray,
        rowsums: np.ndarray,
        colsums: np.ndarray,
        tol: float = 1e-9,
        max_iter: int = 100,
        ) -> np.ndarray:
    """Perform iterative proportional fitting (IPF) in 2D.

    :param values: The seed matrix for IPF. The output matrix will be as close
        as possible to this matrix, given the marginal sums.
    :param rowsums: Target marginal sums for output matrix rows.
    :param colsums: Target marginal sums for output matrix columns.
    :param tol: Tolerance as a termination condition. When the sum of differences
        between two iterations of IPF gets below this threshold, terminate the
        algorithm.
    :param max_iter: Maximum number of iterations. After this many iterations,
        the algorithm will terminate regardless of convergence.
    """
    rowsums = rowsums.reshape(-1, 1)
    colsums = colsums.reshape(1, -1)
    prevalues = values.copy()
    diff = float('inf')
    for i in range(max_iter):
        prevalues[:] = values[:]
        prerowsums = values.sum(axis=1)
        values *= rowsums / np.where(prerowsums == 0, 1, prerowsums).reshape(-1, 1)
        precolsums = values.sum(axis=0)
        values *= colsums / np.where(precolsums == 0, 1, precolsums).reshape(1, -1)
        diff = abs(prevalues - values).sum()
        if diff <= tol:
            prerowsums = values.sum(axis=1)
            values *= rowsums / np.where(prerowsums == 0, 1, prerowsums).reshape(-1, 1)
            break
    if diff <= tol:
        warnings.warn(f'IPF did not converge (step diff sum at termination: {diff})')
    return values


def srid_to_crsdef(srid: int) -> Dict[str, str]:
    """Convert from an EPSG CRS ID to a Proj4 dictionary CRS definition."""
    return {'init' : 'epsg:' + str(srid)}


def proj(crsdef: Dict[str, Any]) -> pyproj.Proj:
    """Create a PyProj CRS object from a Proj4 dictionary CRS definition."""
    return pyproj.Proj(**crsdef)


def srid_proj(srid: int) -> pyproj.Proj:
    """Create a PyProj CRS object from an EPSG CRS ID."""
    return pyproj.Proj(srid)


def transformation(from_proj: pyproj.Proj,
                   to_proj: pyproj.Proj,
                   ) -> Callable[[shapely.geometry.base.BaseGeometry], shapely.geometry.base.BaseGeometry]:
    """Build a projection function from one CRS to the other."""
    projtrans = functools.partial(pyproj.transform, from_proj, to_proj)
    shtrans = shapely.ops.transform
    return lambda geom: shtrans(projtrans, geom)


def point_gdf(df: pd.DataFrame,
              xcol: str = 'X',
              ycol: str = 'Y',
              srid: int = 4326,
              drop_locs: bool = False,
              ) -> gpd.GeoDataFrame:
    """Build a point geodataframe from a dataframe with x- and y-coordinates."""
    return gpd.GeoDataFrame(
        (df.drop([xcol, ycol], axis=1) if drop_locs else df),
        crs=srid_to_crsdef(srid),
        geometry=[
            shapely.geometry.Point(xy)
                if not np.isnan(xy).sum() else shapely.geometry.Point()
            for xy in zip(df[xcol], df[ycol])
        ]
    )


def wkt_gdf(df: pd.DataFrame,
            wkt_col: str = CHECK_GEOM_COLS[0],
            srid: int = 4326,
            drop_locs: bool = True,
            ):
    """Build a geodataframe from a dataframe with WKT."""
    return gpd.GeoDataFrame(
        (df.drop(wkt_col, axis=1) if drop_locs else df),
        crs=srid_to_crsdef(srid),
        geometry=gpd.GeoSeries.from_wkt(df[wkt_col]),
    )


def load_extent(path: os.PathLike,
                target_srid: Optional[int] = None,
                ) -> shapely.geometry.base.BaseGeometry:
    """Load an extent polygon from a file, optionally reprojecting to another CRS.

    Only the first geometry found in the file is used as the extent. If None
    is given as the path, None is returned.
    """
    extent_df = gpd.read_file(path)
    poly = extent_df.loc[extent_df.index[0],'geometry']
    if target_srid:
        target_proj = srid_proj(target_srid)
        source_proj = proj(extent_df.crs)
        if source_proj != target_proj:
            poly = transformation(source_proj, target_proj)(poly)
    return poly


def read_gdf(path: os.PathLike, **kwargs) -> gpd.GeoDataFrame:
    """Read a geodataframe from a file at the given path.

    If the file is a CSV, :func:`read_csv_gdf` is used, otherwise, GDAL machinery
    is invoked through ``geopandas.read_file``.
    """
    if os.fspath(path).endswith('.csv'):
        return read_csv_gdf(path, **kwargs)
    else:
        return gpd.read_file(path)


def write_gdf(gdf: gpd.GeoDataFrame, path: os.PathLike) -> None:
    """Write a geodataframe to a file at the given path.

    If the path denotes a CSV file, ``pandas.DataFrame.to_csv`` is used (which
    converts the geometries to WKT); otherwise, GDAL machinery is invoked
    through ``geopandas.GeoDataFrame.to_file``.
    """
    if path.endswith('.csv'):
        gdf.to_csv(path, sep=';', index=False)
    else:
        gdf.to_file(path)


def read_csv_gdf(path: os.PathLike,
                 srid: int = 4326,
                 check_wkt_cols: List[str] = CHECK_GEOM_COLS,
                 **kwargs) -> gpd.GeoDataFrame:
    """Read a geodataframe from a CSV file at the given path.

    The CSV is loaded using pandas and its columns are inspected. If any
    of ``check_wkt_cols`` is found, the geometry is loaded from that column's
    WKT values, otherwise, :func:`point_gdf` is called on the dataframe
    with any superfluous keyword arguments of this function. srid determines
    the EPSG CRS ID to be assigned to the geometries.
    """
    df = pd.read_csv(path, sep=';')
    cols = [col.lower() for col in df.columns.tolist()]
    for col in check_wkt_cols:
        if col in cols:
            return wkt_gdf(df, df.columns[cols.index(col)], srid=srid)
    else:
        return point_gdf(df, srid=srid, **kwargs)


def read_places(args) -> gpd.GeoDataFrame:
    """A shorthand for read_gdf from commandline arguments."""
    return read_gdf(args.place_file, xcol=args.x_col, ycol=args.y_col, srid=args.srid)


def read_nonspatial(path: os.PathLike, **kwargs) -> pd.DataFrame:
    """Read a nonspatial dataframe from the file at the given path.

    If the file is a CSV, it is loaded via pandas ``read_csv`` with any keyword
    arguments passed to it; otherwise, the file is loaded as a spatial file
    into a geodataframe and its geometry stripped.
    """
    if os.fspath(path).endswith('.csv'):
        return pd.read_csv(path, sep=';', **kwargs)
    else:
        return pd.DataFrame(gpd.read_file(path).drop('geometry', axis=1))


def point_gdf_to_wkt(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Convert a point geodataframe to a saveable dataframe with wkt, x, y columns."""
    return (
        gdf
        .assign(wkt=gdf.geometry.map(lambda x: x.wkt))
        .assign(x=gdf.geometry.x)
        .assign(y=gdf.geometry.y)
        .drop(['geometry'], axis=1)
    )


AnyPolygon = Union[
    shapely.geometry.Polygon,
    shapely.geometry.MultiPolygon,
]