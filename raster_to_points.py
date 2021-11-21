"""Converts raster file(s) to a point layer, optionally clipping to an area."""

import os
import csv
from typing import Tuple, Callable

import geopandas as gpd
import shapely.ops
import shapely.geometry
import shapely.prepared

import mobilib.argparser
import mobilib.core
import mobilib.raster

GeomTransformCallable = Callable[
    [shapely.geometry.base.BaseGeometry],
    shapely.geometry.base.BaseGeometry
]


def load_transformations(extentfile: os.PathLike,
                         src_srid: int,
                         tgt_srid: int,
                         ) -> Tuple[GeomTransformCallable, GeomTransformCallable]:
    if extentfile:
        print('loading extent')
        extent_df = gpd.read_file(extentfile)
        extent = shapely.ops.cascaded_union(extent_df.geometry.values)
        extent_crs = extent_df.crs
    else:
        extent_crs = None
        extent = None
    if tgt_srid:
        outcrs = mobilib.core.srid_proj(tgt_srid)
        if extent_crs and extent_crs != outcrs:
            print('reprojecting extent')
            extent = mobilib.core.transformation(
                mobilib.core.proj(extent_crs), outcrs
            )(extent)
    elif extent_crs:
        outcrs = extent_crs
    else:
        outcrs = None
    if outcrs and src_srid:
        print('preparing coordinate transformation')
        main_trans = mobilib.core.transformation(mobilib.core.srid_proj(src_srid), outcrs)
    else:
        main_trans = lambda x: x
    if extent:
        print('setting up extent filter')
        extent_prep = shapely.prepared.prep(extent)
        filter = lambda geom: extent_prep.contains(geom)
    else:
        filter = lambda geom: True
    return main_trans, filter


parser = mobilib.argparser.default(__doc__)
parser.add_argument(
    'inraster', nargs='+',
    help='input raster files with an associated world file'
)
parser.add_argument(
    'outfile',
    help='path to output semicolon-delimited CSV file'
)
parser.add_argument(
    '-e', '--extent',
    help='GDAL-compatible polygon file to which to clip the output'
)
parser.add_argument(
    '-n', '--nodata', nargs='+', type=int,
    help='NoData value of the rasters (will be ignored)'
)
parser.add_argument(
    '-s', '--source-srid', type=int,
    help='EPSG ID of the coordinate system of the input rasters'
)
parser.add_argument(
    '-r', '--target-srid', type=int,
    help='EPSG ID of the coordinate system to use on output'
)

if __name__ == '__main__':
    args = parser.parse_args()
    reproj, extent_filter = load_transformations(args.extent, args.source_srid, args.target_srid)
    print('opening output', args.outfile)
    with open(args.outfile, 'w', newline='') as outfile:
        wr = csv.writer(outfile, delimiter=';')
        wr.writerow(('x', 'y', 'value'))
        for raster in args.inraster:
            print('processing', raster)
            for ptarr, value in mobilib.raster.load_points(raster, nodata=args.nodata):
                pt = reproj(shapely.geometry.Point(ptarr))
                if extent_filter(pt):
                    wr.writerow((pt.x, pt.y, value))
