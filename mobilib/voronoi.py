
import operator

import numpy
import scipy.spatial
import shapely.geometry
import shapely.geometry.base
import shapely.prepared


def pointset_bounds(coords):
    return (
        min(coords, key=operator.itemgetter(0))[0],
        min(coords, key=operator.itemgetter(1))[1],
        max(coords, key=operator.itemgetter(0))[0],
        max(coords, key=operator.itemgetter(1))[1],
    )


def bounds_to_limiting_generators(minx, miny, maxx, maxy):
    addx = maxx - minx
    addy = maxy - miny
    return [
        (minx - addx, miny - addy),
        (maxx + addx, miny - addy),
        (minx - addx, maxy + addy),
        (maxx + addx, maxy + addy),
    ]


def cells(points, extent=None):
    if extent is None:
        bbox = pointset_bounds(points)
        extent_prep = None
    else:
        if not isinstance(extent, shapely.geometry.base.BaseGeometry):
            extent = shapely.geometry.box(*extent)
        bbox = extent.bounds
        extent_prep = shapely.prepared.prep(extent)
    boundgens = bounds_to_limiting_generators(*bbox)
    diagram = scipy.spatial.Voronoi(numpy.concatenate((points, boundgens)))
    for reg_i in diagram.point_region[:-len(boundgens)]:
        coords = diagram.vertices[diagram.regions[reg_i]]
        poly = shapely.geometry.Polygon(coords)
        if extent_prep is None or extent_prep.contains(poly):
            yield poly
        else:
            yield extent.intersection(poly)


def cells_shapely(points, extent=None):
    return cells(numpy.array([pt.coords[0] for pt in points]), extent=extent)