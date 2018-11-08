
import os

import numpy
import imageio


class World:
    def __init__(self, xcell, yrot, xrot, ycell, xorigin, yorigin):
        self.xcell = xcell
        self.yrot = yrot
        self.xrot = xrot
        self.ycell = ycell
        self.xorigin = xorigin
        self.yorigin = yorigin
        self.matrix = numpy.array((
            (self.yrot, self.xcell),
            (self.ycell, self.xrot)
        ))
        self.shift = numpy.array((self.xorigin + self.xcell / 2, self.yorigin + self.ycell / 2))

    @classmethod
    def from_file(cls, file):
        return cls(*(float(line.strip()) for line in file.readlines()))

    def raster_to_points(self, raster, nodata=[]):
        nodata = set(nodata)
        for pos, value in numpy.ndenumerate(raster):
            if value not in nodata:
                yield self.matrix.dot(pos) + self.shift, value


def load(path, **kwargs):
    return imageio.imread(path, **kwargs)


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
