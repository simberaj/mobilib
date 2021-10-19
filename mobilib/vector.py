import math
from typing import Union

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely.geometry


def length(vectors):
    return np.linalg.norm(vectors, axis=-1)

    
def angle(vectors):
    if len(vectors.shape) == 2:
        return np.arctan2(*vectors.T[::-1])
    else:
        slicer = (slice(None), ) * (len(vectors.shape) - 1)
        return np.arctan2(vectors[slicer + (1,)], vectors[slicer + (0,)])


ALPHA = math.radians(30)
N_PTS = 16
ALPHA_ANGLES = np.linspace(0, ALPHA, N_PTS)


def centroid(ptser: gpd.GeoSeries) -> shapely.geometry.Point:
    return shapely.geometry.MultiPoint(ptser.values).centroid


def straight_line(from_pt: shapely.geometry.Point,
                  to_pt: shapely.geometry.Point,
                  ) -> shapely.geometry.LineString:
    return shapely.geometry.LineString((from_pt, to_pt))


def curved_line(from_pt: shapely.geometry.Point,
                to_pt: shapely.geometry.Point,
                ) -> shapely.geometry.LineString:
    xa = from_pt.x
    ya = from_pt.y
    xb = to_pt.x
    yb = to_pt.y
    if xb == xa:
        if yb == ya:
            return shapely.geometry.LineString((from_pt, to_pt))
        phi = .5 * ALPHA
        r = (yb - ya) / (math.sin(phi - ALPHA) - math.sin(phi))
    else:
        q = (yb - ya) / (xb - xa)
        phi = .5 * (ALPHA + 4 * math.atan(q + math.sqrt(q ** 2 + 1)))
        r = (xb - xa) / (math.cos(phi - ALPHA) - math.cos(phi))
    xs = xa - r * math.cos(phi)
    ys = ya - r * math.sin(phi)
    angles = phi - ALPHA_ANGLES
    x = r * np.cos(angles) + xs
    y = r * np.sin(angles) + ys
    return shapely.geometry.LineString(tuple(zip(x, y)))


def succession_geoms(assignments: pd.Series,
                     attrs: Union[pd.Series, pd.DataFrame],
                     unit_geoms: gpd.GeoSeries,
                     ) -> gpd.GeoDataFrame:    # geometry, weight
    return (
        pd.DataFrame({'region': assignments})
        .join(unit_geoms.rename('geometry'))
        .groupby('region')[['geometry']].agg(shapely.ops.unary_union)
        .join(attrs)
        .rename_axis('id')
    )
