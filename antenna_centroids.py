"""Compute centroids of antenna coverage areas (cells)."""

import argparse

import numpy
import pandas as pd
import geopandas as gpd
import shapely.geometry
import matplotlib.patches
import matplotlib.pyplot as plt

import mobilib.voronoi

parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('infile',
    help='input antenna locations as semicolon-delimited CSV'
)
parser.add_argument('outfile',
    help='path to output the centroid CSV'
)
parser.add_argument('-e', '--extent',
    help='GDAL-compatible polygon file to which to clip the output'
)
parser.add_argument('-x', '--xcol', default='x',
    help='name of the X-coordinate column in input'
)
parser.add_argument('-y', '--ycol', default='y',
    help='name of the Y-coordinate column in input'
)
parser.add_argument('-i', '--idcol', default='cell_name',
    help='name of antenna identifier column (unique)'
)
parser.add_argument('-a', '--azimuthcol', default='azim',
    help='name of antenna orientation azimuth column (in degrees)'
)
parser.add_argument('-w', '--weighting',
    help='CSV file with centroid-weighting points (x, y and optional weight column)'
)
parser.add_argument('-W', '--weightcol',
    help='weight column name in centroid-weighting point table'
)

X_MODIF_COL = 'xmod'
Y_MODIF_COL = 'ymod'
X_CENT_COL = 'xcent'
Y_CENT_COL = 'ycent'
X_WT_COL = 'xweighted'
Y_WT_COL = 'yweighted'

if __name__ == '__main__':
    args = parser.parse_args()
    print('loading antennas')
    antennas = pd.read_csv(args.infile, sep=';')
    if args.extent:
        print('loading extent')
        extent_gdf = gpd.read_file(args.extent)
        extent = extent_gdf.loc[extent_gdf.index[0],'geometry']
    else:
        extent = None
    # todo check with erki what those 32767 azimuths mean
    print('computing antenna locations')
    antennas = antennas[antennas[args.xcol] > 0]
    azimuths = antennas[args.azimuthcol]
    antennas.loc[azimuths > 360, args.azimuthcol] = numpy.nan
    antennas[X_MODIF_COL] = (
        antennas[args.xcol]
        + numpy.cos(numpy.radians(azimuths.fillna(90)))
    )
    antennas[Y_MODIF_COL] = (
        antennas[args.ycol]
        + numpy.sin(numpy.radians(azimuths.fillna(0)))
    )
    sites = antennas[[X_MODIF_COL, Y_MODIF_COL]].drop_duplicates().reset_index().rename(columns={'index' : 'site_i'})
    print('computing antenna cells')
    sitegeoms = list(mobilib.voronoi.cells(
        numpy.stack((sites[X_MODIF_COL], sites[Y_MODIF_COL])).transpose(),
        extent=extent
    ))
    cells = gpd.GeoDataFrame({'site_i' : sites.site_i}, geometry=sitegeoms)
    print('computing antenna centroids')
    ord_centx, ord_centy = numpy.array(list(zip(*(
        cell.centroid.coords[0] for cell in sitegeoms
    ))))
    if args.weighting:
        print('loading weighting points')
        weight_df = pd.read_csv(args.weighting, sep=';')
        weightcol = args.weightcol if args.weightcol else 'weight'
        weight_gdf = gpd.GeoDataFrame(weight_df, geometry=[
            shapely.geometry.Point(*xy) for xy in zip(weight_df.x, weight_df.y)
        ])
        weight_gdf[X_WT_COL] = weight_gdf.x * weight_gdf[weightcol]
        weight_gdf[Y_WT_COL] = weight_gdf.y * weight_gdf[weightcol]
        if weightcol not in weight_gdf:
            weight_gdf[weightcol] = 1
        print('assigning weighting points to cells')
        weight_gdf_joined = gpd.sjoin(weight_gdf, cells, how='right', op='within')
        print('weighting centroids')
        centroids = weight_gdf_joined.groupby('site_i').agg({
            X_WT_COL : numpy.sum,
            Y_WT_COL : numpy.sum,
            weightcol : numpy.sum,
        }).reset_index()
        empty_cells = (centroids[weightcol] == 0)
        centroids[X_CENT_COL] = numpy.where(
            empty_cells, ord_centx,
            centroids[X_WT_COL] / centroids[weightcol]
        )
        centroids[Y_CENT_COL] = numpy.where(
            empty_cells, ord_centy,
            centroids[Y_WT_COL] / centroids[weightcol]
        )
        sites = pd.merge(
            sites, centroids[['site_i', X_CENT_COL, Y_CENT_COL]],
            on='site_i'
        )
    else:
        sites[X_CENT_COL], sites[Y_CENT_COL] = ord_centx, ord_centy
    antennas = pd.merge(antennas, sites, on=(X_MODIF_COL, Y_MODIF_COL))

    plt.scatter(antennas[X_MODIF_COL], antennas[Y_MODIF_COL], s=9, color='blue')
    plt.scatter(antennas[X_CENT_COL], antennas[Y_CENT_COL], s=9, color='red')
    ax = plt.gca()
    # for cell in sitegeoms:
        # if isinstance(cell, shapely.geometry.Polygon):
            # cell = shapely.geometry.MultiPolygon([cell])
        # for part in cell:
            # ax.add_patch(matplotlib.patches.Polygon(
                # numpy.array(part.exterior),
                # lw=0.25, facecolor=None, edgecolor='#bbbbbb'
            # ))
        # # plt.plot(*cell.exterior.xy, color='#bbbbbb', lw=0.25)
    ax.set_aspect('equal')
    plt.show()

    antennas.to_csv(args.outfile, sep=';', index=False)

