
'''Generic, all-purpose functions.'''

import functools

import pyproj
import shapely.ops


def proj(crsdef):
    return pyproj.Proj(**crsdef)

def srid_proj(srid):
    return pyproj.Proj(init='epsg:' + str(srid))
    
    
def transformation(from_proj, to_proj):
    projtrans = functools.partial(pyproj.transform, from_proj, to_proj)
    shtrans = shapely.ops.transform
    return lambda geom: shtrans(projtrans, geom)