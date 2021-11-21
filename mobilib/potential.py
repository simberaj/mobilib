"""Compute geographic potential across space."""

from typing import Tuple

import numpy as np
import pandas as pd

from . import raster as mobilib_raster


class Decayer:
    def __init__(self,
                 halfrange: float,
                 method: str = 'gaussian',
                 lag: float = 0
                 ):
        self.method = method
        self.decay = self.METHODS[self.method](halfrange - lag)
        self.lag = None if lag == 0 else DistanceLag(lag)

    def apply(self, distance):
        if self.lag:
            distance = self.lag.apply(distance)
        return self.decay.apply(distance)

    def invert(self, weight):
        distance = self.decay.invert(weight)
        if self.lag:
            distance = self.lag.invert(distance)
        return distance


class DistanceLag:
    def __init__(self, lag: float = 0):
        self.lag = lag

    def apply(self, distance):
        if self.lag == 0:
            return distance
        else:
            raw = distance - self.lag
            return np.where(raw < 0, 0, raw)

    def invert(self, distance):
        return distance + self.lag


class GaussianDecay:
    LOG2 = np.log(2)

    def __init__(self, halfrange: float):
        self.halfrange = halfrange
        self.halfrange_sq = self.halfrange ** 2

    def apply(self, distance):
        return 2 ** (-distance ** 2 / self.halfrange_sq)

    def invert(self, weight):
        return self.halfrange * np.sqrt(-np.log(weight) / self.LOG2)


class CauchyDecay:
    def __init__(self, halfrange: float):
        self.halfrange = halfrange
        self.halfrange_sq = self.halfrange ** 2

    def apply(self, distance):
        return self.halfrange_sq / (self.halfrange_sq + distance ** 2)

    def invert(self, weight):
        return self.halfrange * np.sqrt((1 - weight) / weight)


class LinearDecay:
    def __init__(self, halfrange: float):
        self.halfrange = halfrange

    def apply(self, distance):
        decayed = 1 - distance / (2 * self.halfrange)
        return np.where(decayed < 0, 0, decayed)

    def invert(self, weight):
        distance = 2 * self.halfrange * (1 - weight)
        return np.where(distance < 0, 0, distance)


Decayer.METHODS = {
    'gaussian': GaussianDecay,
    'linear': LinearDecay,
    'cauchy': CauchyDecay,
}


def raster(data_df: pd.DataFrame,
           cell_size: float = 1,
           decay: str = 'gaussian',
           mode: str = 'sum',
           tolerance: float = 1e-7,
           ) -> Tuple[np.ndarray, mobilib_raster.World]:
    xmin, ymin, xmax, ymax = mobilib_raster.calculate_bounds(
        data_df.x, data_df.y, data_df.halfrange.max().item(), extension=cell_size
    )
    cell_df = data_df.copy()
    for coor_col, min_dim in (('x', xmin), ('y', ymin)):
        cell_df[coor_col] -= min_dim
    for col in cell_df.columns:
        if col != 'magnitude':
            cell_df[col] /= cell_size
    nrows = int(round((ymax - ymin) / cell_size))
    ncols = int(round((xmax - xmin) / cell_size))
    rowrange = np.arange(nrows) + .5
    colrange = np.arange(ncols) + .5
    potential = np.zeros((nrows, ncols))
    i = 0
    for x, y, magnitude, halfrange, fullrange in cell_df.itertuples(index=False, name=None):
        if magnitude <= 0: continue
        decayer = Decayer(halfrange, decay, fullrange)
        spread = decayer.invert(tolerance / magnitude)
        rowfrom, colfrom = int(np.floor(y - spread)), int(np.floor(x - spread))
        rowto, colto = int(np.ceil(y + spread)), int(np.ceil(x + spread))
        distmat = np.sqrt(
            (y - rowrange[rowfrom:rowto]).reshape(-1, 1) ** 2
            + (x - colrange[colfrom:colto]) ** 2
        )
        weights = decayer.apply(distmat) * magnitude
        if mode == 'sum':
            potential[rowfrom:rowto,colfrom:colto] += weights
        elif mode == 'max':
            potential[rowfrom:rowto,colfrom:colto] = np.where(
                potential[rowfrom:rowto,colfrom:colto] < weights,
                weights,
                potential[rowfrom:rowto,colfrom:colto]
            )
        else:
            raise NotImplementedError('unknown mode: ' + mode)
        i += 1
        # print(i, end=' \r')
    return potential[::-1], mobilib_raster.World.create_rect(xmin, ymax, cell_size)
