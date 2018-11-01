
'''Computes antenna centroids.'''

import sys

import numpy
import pandas as pd
import geopandas as gpd
import shapely.geometry
import matplotlib.patches
import matplotlib.pyplot as plt

import mobilib.voronoi


if __name__ == '__main__':
    antennas = pd.read_csv(sys.argv[1], sep=';')
    extent_gdf = gpd.read_file(sys.argv[2])
    extent = extent_gdf.loc[extent_gdf.index[0],'geometry']
    # todo check with erki what those 32767 azimuths mean
    antennas = antennas[antennas['x'] > 0]
    antennas.loc[antennas['azim']>360,'azim'] = numpy.nan
    antennas['xmod'] = antennas['x'] + numpy.cos(numpy.radians(antennas['azim'].fillna(90)))
    antennas['ymod'] = antennas['y'] + numpy.sin(numpy.radians(antennas['azim'].fillna(0)))
    sites = antennas.groupby(['xmod', 'ymod']).agg({'cell_name' : list}).reset_index()
    sitegeoms = list(mobilib.voronoi.cells(
        numpy.stack((sites['xmod'], sites['ymod'])).transpose(),
        extent
    ))
    sites['xcent'], sites['ycent'] = zip(*(cell.centroid.coords[0] for cell in sitegeoms))
    antennas = pd.merge(antennas, sites.drop(['cell_name'], axis=1), on=('xmod', 'ymod'))

    plt.scatter(antennas.xcent, antennas.ycent, s=9)
    ax = plt.gca()
    for cell in sitegeoms:
        if isinstance(cell, shapely.geometry.Polygon):
            cell = shapely.geometry.MultiPolygon([cell])
        for part in cell:
            ax.add_patch(matplotlib.patches.Polygon(
                numpy.array(part.exterior),
                lw=0.25, facecolor=None, edgecolor='#bbbbbb'
            ))
        # plt.plot(*cell.exterior.xy, color='#bbbbbb', lw=0.25)
    ax.set_aspect('equal')
    plt.show()

    antennas.to_csv(sys.argv[3], sep=';', index=False)

