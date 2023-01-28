'''Assign speeds to roads and compute their lengths and travel times.'''

import os
import json
import argparse
from typing import Any, List, Dict, Callable, Iterable, Generator, Optional

import shapely.strtree
import fiona.crs
import fiona.transform
import geopandas as gpd


def load_urban_areas(urban_file: os.PathLike,
                     crs: Dict[str, Any],
                     ) -> shapely.strtree.STRtree:
    gs = gpd.read_file(urban_file).geometry
    if gs.crs != crs:
        gs = gs.to_crs(crs=crs)
    return shapely.strtree.STRtree(gs.values)


def reproject(features: Iterable[Dict[str, Any]],
              src_crs: Dict[str, Any],
              tgt_crs: Dict[str, Any],
              ) -> Generator[Dict[str, Any], None, None]:
    if src_crs != tgt_crs:
        for feat in features:
            yield {
                'type': 'Feature',
                'properties': feat['properties'],
                'geometry': fiona.transform.transform_geom(
                    src_crs, tgt_crs, feat['geometry']
                )
            }
    else:
        yield from features


def intersect_urban(roads: Iterable[Dict[str, Any]],
                    urban_register: Optional[shapely.strtree.STRtree] = None,
                    ) -> Generator[Dict[str, Any], None, None]:
    if urban_register is None:
        for feat in roads:
            feat['properties']['urban'] = False
            yield feat
    else:
        for feat in roads:
            props = feat['properties']
            main_shape = shapely.geometry.shape(feat['geometry'])
            urban_shapes = []
            nonurban_shapes = [main_shape]
            for urban_chunk in urban_register.query(main_shape):
                for shape in nonurban_shapes:
                    if urban_chunk.intersects(shape):
                        nonurban_shapes.remove(shape)
                        if urban_chunk.contains(shape):
                            urban_shapes.append(shape)
                        else:
                            urbans, nonurbans = split_line(shape, urban_chunk)
                            urban_shapes.extend(urbans)
                            nonurban_shapes.extend(nonurbans)
                if not nonurban_shapes:
                    break
            all_shapes = urban_shapes + nonurban_shapes
            urban_flags = [True] * len(urban_shapes) + [False] * len(nonurban_shapes)
            for shape, urban_flag in zip(all_shapes, urban_flags):
                feat_props = props.copy()
                feat_props['urban'] = urban_flag
                yield {
                    'type': 'Feature',
                    'properties': feat_props,
                    'geometry': shapely.geometry.mapping(shape)
                }


def split_line(line, polygon):
    inside = []
    outside = []
    for geom in shapely.ops.split(line, polygon).geoms:
        if geom.intersection(polygon).length > .95 * geom.length:
            inside.append(geom)
        else:
            outside.append(geom)
    return inside, outside



def speedify_props(props: Dict[str, Any],
                   geom: Dict[str, Any],
                   speed_func: Callable[[Dict[str, Any]], int]
                   ) -> None:
    props['speed'] = speed_func(props)                        # in kph
    props['length'] = shapely.geometry.asShape(geom).length   # in m
    props['time'] = props['length'] / props['speed'] * 3.6    # in s


def create_speed_func(config: Dict[str, Any]) -> Callable[[Dict[str, Any]], int]:
    mapper = create_speed_mapper(config['mapping'])
    maxprop = config['maxprop']['attribute']
    def speed_fx(props, mapper=mapper):
        prop_val = props.get(mapper['attribute'])
        after_map = mapper['dict'].get(prop_val)
        if after_map is None:
            return mapper['default']
        elif isinstance(after_map, dict):
            return speed_fx(props, mapper=after_map)
        else:
            maxspeed = props.get(maxprop, 0)
            return min(after_map, maxspeed) if maxspeed else after_map
    return speed_fx


def create_speed_mapper(config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'attribute': config['attribute'],
        'default': config['default'],
        'dict': dict(zip(config['keys'], [
            create_speed_mapper(val) if isinstance(val, dict) else val for val in config['values']
        ]))
    }


def speedify_roads(source_file: os.PathLike,
                   target_file: os.PathLike,
                   config_file: os.PathLike,
                   epsg_id: int,
                   urban_file: Optional[os.PathLike] = None,
                   ) -> None:
    with open(config_file) as conffile:
        config = json.load(conffile)
    crs = fiona.crs.from_epsg(epsg_id)
    if urban_file is not None:
        urban_register = load_urban_areas(urban_file, crs)
    else:
        urban_register = None
    speed_func = create_speed_func(config['road']['speed'])
    i = 0
    with fiona.Env():
        with fiona.open(source_file, 'r') as src:
            schema = src.schema.copy()
            schema['properties'].update({
                'urban': 'bool',
                'speed': 'int',
                'length': 'float',
                'time': 'float',
            })
            with fiona.open(target_file, 'w', driver=src.driver, crs=crs, schema=schema) as tgt:
                for feat in intersect_urban(reproject(src, src.crs, crs), urban_register):
                    speedify_props(feat['properties'], feat['geometry'], speed_func)
                    tgt.write(feat)
                    i += 1
                    if i % 100 == 0:
                        print(i, end='\r')
                    # if i > 1000:
                        # break



parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('in_roads',
    help='GDAL-compatible spatial file with land use polygons'
)
parser.add_argument('out_roads',
    help='path to output a GDAL-compatible spatial file with annotated roads'
)
parser.add_argument('-u', '--urban-areas',
    help='a GDAL-compatible spatial file with urban area layer'
)
parser.add_argument('-c', '--conf', default='config.json',
    help='JSON config file containing road speeds'
)
parser.add_argument('-s', '--crs', default=3035,
    help='EPSG ID of the planar CRS to use'
)

if __name__ == '__main__':
    args = parser.parse_args()
    speedify_roads(
        args.in_roads,
        args.out_roads,
        args.conf,
        args.crs,
        urban_file=args.urban_areas
    )