'''Create a urban area layer from the OSM land use layer.'''

import os
import json
import argparse
from typing import Any, Dict, Set

import fiona
import geopandas as gpd
import shapely.geometry


def load_urban_lu_geoms(source_file: os.PathLike,
                        filters: Dict[str, Set[Any]],
                        ) -> gpd.GeoSeries:
    geoms = []
    with fiona.Env():
        with fiona.open(source_file, 'r') as src:
            for feat in src:
                feat_props = feat['properties']
                for prop_name, allowed_list in filters.items():
                    if feat_props.get(prop_name) not in allowed_list:
                        break
                else:
                    geoms.append(shapely.geometry.shape(feat['geometry']))
    return gpd.GeoSeries(geoms, crs={'init': 'epsg:4326', 'no_defs': True})


def create_urban_layer(source_file: os.PathLike,
                       target_file: os.PathLike,
                       config_file: os.PathLike,
                       epsg_id: int,
                       ) -> None:
    with open(config_file) as conffile:
        config = json.load(conffile)
    urban_lu_geoms = load_urban_lu_geoms(
        source_file, config['landuse']['urban']['filter']
    ).to_crs(
        epsg=epsg_id
    ).buffer(
        distance=config['landuse']['urban']['buffer'],
        resolution=4,
    )
    unionized = shapely.ops.unary_union(urban_lu_geoms.values).geoms
    out_gs = gpd.GeoSeries([
        geom for geom in unionized if geom.area > config['landuse']['urban']['minsize']
    ], crs={'init': 'epsg:' + str(epsg_id), 'no_defs': True})
    gpd.GeoDataFrame(
        {'i': range(len(out_gs))},
        geometry=out_gs
    ).to_file(target_file)


parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('in_landuse',
    help='GDAL-compatible spatial file with land use polygons'
)
parser.add_argument('out_urban',
    help='path to output a GDAL-compatible spatial file with dissolved urban area layer'
)
parser.add_argument('-c', '--conf', default='config.json',
    help='JSON config file containing urban land use classes'
)
parser.add_argument('-s', '--crs', default=3035,
    help='EPSG ID of the planar CRS to use'
)

if __name__ == '__main__':
    args = parser.parse_args()
    create_urban_layer(args.in_landuse, args.out_urban, args.conf, args.crs)