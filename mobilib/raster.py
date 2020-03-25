
import os
from typing import Tuple

import numpy as np
import gdal
import osr


class World:
    def __init__(self, xcell, yrot, xrot, ycell, xorigin, yorigin):
        self.xcell = xcell
        self.yrot = yrot
        self.xrot = xrot
        self.ycell = ycell
        self.xorigin = xorigin
        self.yorigin = yorigin
        self.matrix = np.array((
            (self.yrot, self.xcell),
            (self.ycell, self.xrot)
        ))
        self.shift = np.array((self.xorigin + self.xcell / 2, self.yorigin + self.ycell / 2))

    @classmethod
    def from_file(cls, file):
        return cls(*(float(line.strip()) for line in file.readlines()))

    def raster_to_points(self, raster, nodata=[]):
        nodata = set(nodata)
        for pos, value in np.ndenumerate(raster):
            if value not in nodata:
                yield self.matrix.dot(pos) + self.shift, value

    def to_gdal_tuple(self) -> Tuple[float, ...]:
        return (self.xorigin, self.xcell, self.xrot, self.yorigin, self.yrot, self.ycell)

    @classmethod
    def create_rect(cls, xmin, ymin, cell_size):
        return cls(cell_size, 0, 0, -cell_size, xmin, ymin)


def load_points(path, worldpath=None, nodata=None, **kwargs):
    if worldpath is None:
        worldpath = find_world_path(path)
    with open(worldpath, 'r') as worldfile:
        world = World.from_file(worldfile)
    return world.raster_to_points(load(path, **kwargs), nodata=nodata)


def find_world_path(impath):
    name, ext = os.path.splitext(impath)
    for worldext in possible_world_extensions(ext):
        worldname = name + worldext
        if os.path.exists(worldname):
            return worldname
    raise FileNotFoundError('world file not found for ' + impath)


def possible_world_extensions(ext):
    yield ext[:2] + ext[-1] + 'w'
    yield ext + 'w'
    yield '.wld'


def calculate_bounds(xs, ys, cell_size, extension=0):
    return (
        ((xs.min().item() - extension) // cell_size) * cell_size,
        ((ys.min().item() - extension) // cell_size) * cell_size,
        ((xs.max().item() + extension) // cell_size + 1) * cell_size,
        ((ys.max().item() + extension) // cell_size + 1) * cell_size,
    )

def to_geotiff(array: np.ndarray,
               path: str,
               world: World,
               srid: int = 4326
               ) -> None:
    assert array.ndim == 2
    cols = array.shape[1]
    rows = array.shape[0]
    out_raster = gdal.GetDriverByName('GTiff').Create(
        path, cols, rows, 1, gdal_raster_type(array.dtype.type)
    )
    out_raster.SetGeoTransform(world.to_gdal_tuple())
    outband = out_raster.GetRasterBand(1)
    outband.WriteArray(array)
    osr_crs = osr.SpatialReference()
    osr_crs.ImportFromEPSG(srid)
    out_raster.SetProjection(osr_crs.ExportToWkt())
    outband.FlushCache()

GDAL_TYPES = {
    np.float32: gdal.GDT_Float32,
    np.float64: gdal.GDT_Float64,
}

def gdal_raster_type(dtype):
    return GDAL_TYPES[dtype]
