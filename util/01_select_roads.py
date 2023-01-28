'''Select rideable roads and their properties important for routing.'''

import sys
import os
import json
import argparse
from typing import Any, Dict, List, Set, Iterable, Collection, Generator

import fiona
import fiona.crs

def filter_roads(gener: Iterable[dict],
                 filters: Dict[str, Set[Any]]
                 ) -> Generator[dict, None, None]:
    for feat in gener:
        feat_props = feat['properties']
        for prop_name, allowed_list in filters.items():
            if feat_props.get(prop_name) not in allowed_list:
                break
        else:
            yield feat


def select_road_props(props: Dict[str, Any],
                      keep_list: Collection[str]
                      ) -> Dict[str, Any]:
    return {key: val for key, val in props.items() if key in keep_list}


def repair_default_props(props: Dict[str, Any],
                         defaults: Dict[str, Dict[str, Dict[str, Any]]],
                         ) -> Dict[str, Any]:
    for repair_key, repair_sources in defaults.items():
        if repair_key in props:
            for source_key, source_values in repair_sources.items():
                if source_key in props:
                    repair_value = source_values.get(props[source_key])
                    if repair_value is not None:
                        props[repair_key] = repair_value


def create_filtered_roads(source_file: os.PathLike,
                          target_file: os.PathLike,
                          config_file: os.PathLike
                          ) -> None:
    with open(config_file) as conffile:
        config = json.load(conffile)
    with fiona.Env():
        with fiona.open(source_file, 'r') as src:
            out_schema = src.schema.copy()
            out_schema['properties'] = select_road_props(
                out_schema['properties'],
                config['road']['properties']
            )
            crs = src.crs
            if 'init' in crs:
                crs = fiona.crs.from_epsg(crs['init'].strip('epsg:'))
            default_conf = config['road']['defaults']
            with fiona.open(target_file, 'w', driver=src.driver, crs=crs, schema=out_schema) as tgt:
                i = 0
                for feat in filter_roads(src, config['road']['rideable']):
                    feat['properties'] = select_road_props(
                        feat['properties'],
                        config['road']['properties']
                    )
                    repair_default_props(feat['properties'], default_conf)
                    tgt.write(feat)
                    i += 1
                    if i % 1000 == 0:
                        print(i, end='\r')
            print()


parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('in_roads',
    help='GDAL-compatible spatial file with roads'
)
parser.add_argument('out_roads',
    help='path to output a GDAL-compatible spatial file with selected (rideable) roads'
)
parser.add_argument('-c', '--conf', default='config.json',
    help='JSON config file containing road rideability criteria and properties to retain in the "road" key'
)

if __name__ == '__main__':
    args = parser.parse_args()
    create_filtered_roads(args.in_roads, args.out_roads, args.conf)