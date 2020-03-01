import os
import argparse

import numpy as np
import pandas as pd
import geopandas as gpd

import mobilib
import mobilib.potential
import mobilib.argparser

        
parser = mobilib.argparser.default(__doc__, places=True, add_places_id=False)
parser.add_argument('output_raster',
    help='path to output a GeoTIFF antipotential raster'
)
parser.add_argument('-m', '--magnitude-col',
    help='magnitude field in places with (defaults to 1 everywhere if not given)'
)
parser.add_argument('-d', '--decay-function', default='gaussian',
    help='type of distance decay function to use for potential decay'
)
parser.add_argument('-r', '--halfrange-col',
    help='field in supply points with the distance at which half of potential is gone'
)
parser.add_argument('-R', '--halfrange', default=10000, type=float,
    help='constant distance at which half of potential is gone'
)
parser.add_argument('-f', '--fullrange-col',
    help='field in supply points with the distance up to which potential is full'
)
parser.add_argument('-F', '--fullrange', default=400, type=float,
    help='constant distance up to which potential is full'
)
parser.add_argument('-s', '--cell-size', default=1000, type=float,
    help='cell size of the output antipotential raster'
)
parser.add_argument('-C', '--output-srid', default=4326, type=int,
    help='EPSG SRID for output raster'
)


if __name__ == '__main__':
    args = parser.parse_args()
    places_gdf = mobilib.read_places(args)
    # TODO create potential raster
    # then potential subtraction script that will remove a fraction of the
    # potential
    # demand_gdf = gpd.read_file(args.demand_points)
    
    # if args.demand_magnitude_fld is None:
        # supply_all_gdf['magnitude'] = 1
    # else:
        # supply_all_gdf['magnitude'] = supply_all_gdf[args.demand_magnitude_fld]
    # supply_all_gdf = 