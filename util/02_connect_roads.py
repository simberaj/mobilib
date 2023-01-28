'''Connect (cut) roads so that they only touch at endpoints.'''

import sys
import os
import argparse
from typing import Any, Dict, List, Set, Iterable, Collection, Generator, Optional, Tuple

import fiona
import shapely.geometry
import shapely.strtree
import shapely.ops
import shapely.prepared


DEFAULT_LAYER = 0


def read_splitters(source_file: os.PathLike,
                   layer_fld: Optional[str] = None,
                   endpoints_only: bool = False
                   ) -> Tuple[shapely.strtree.STRtree, Dict[int, shapely.strtree.STRtree]]:
    endpoints = set()
    lines = {}
    with fiona.Env():
        with fiona.open(source_file, 'r') as src:
            i = 0
            for feat in src:
                geom = feat['geometry']
                layer = feat['properties'].get(layer_fld, DEFAULT_LAYER)
                endpoints.add(tuple(geom['coordinates'][0]))
                endpoints.add(tuple(geom['coordinates'][-1]))
                if not endpoints_only:
                    lines.setdefault(layer, []).append(shapely.geometry.shape(geom))
                i += 1
                if i % 1000 == 0:
                    print(i, end='\r')
                # if geom['type'] == 'LineString':
                    # if endpoints_only:
                        # splitters.append(shapely.geometry.Point(*geom['coordinates'][0]))
                        # splitters.append(shapely.geometry.Point(*geom['coordinates'][-1]))
                    # else:
                        # splitters.append()
                # else:
                    # raise ValueError(f'unknown geometry: {geom}')
    return (
        shapely.strtree.STRtree([
            shapely.geometry.Point(*pt) for pt in endpoints
        ]),
        {
            layer: shapely.strtree.STRtree(geoms)
            for layer, geoms in lines.items()
        }
    )


def split_lines(source_file: os.PathLike,
                endpoints: shapely.strtree.STRtree,
                lines: Dict[int, shapely.strtree.STRtree],
                layer_fld: Optional[str],
                target_file: os.PathLike,
                ) -> None:
    i = 0
    # print(endpoints._n_geoms, 'endpoints in tree')
    # for lid, ltree in lines.items():
        # print(ltree._n_geoms, 'lines in layer', lid)
    with fiona.Env():
        with fiona.open(source_file, 'r') as src:
            with fiona.open(target_file, 'w', **src.meta) as tgt:
                for src_feat in src:
                    i += 1
                    for split_feat in split_line(src_feat, endpoints, lines, layer_fld):
                        tgt.write(split_feat)
                    if i % 100 == 0:
                        print(i, end='\r')


# def split_line_by_points(geom: shapely.geometry.linestring.LineString,
                         # splitters: List[shapely.geometry.point.Point],
                         # check_intersection: bool = True,
                         # ) -> Iterable[shapely.geometry.linestring.LineString]:
    # endpoints = geom.boundary.geoms
    # splitters = [spl for spl in splitters if spl not in endpoints]
    # if check_intersection:
        # if not splitters:
            # return [geom]
        # geom_checker = shapely.prepared.prep(geom) if len(splitters) > 10 else geom
        # splitters = [spl for spl in splitters if geom_checker.intersects(spl)]
    # if not splitters:
        # return [geom]
    # elif len(splitters) == 1:
        # return shapely.ops.split(geom, splitters[0])
    # else:
        # return shapely.ops.split(geom, shapely.geometry.MultiPoint(splitters))


# def split_line_by_lines(geom: shapely.geometry.linestring.LineString,
                        # splitters: List[shapely.geometry.linestring.LineString],
                        # ) -> Iterable[shapely.geometry.linestring.LineString]:
    # geom_checker = shapely.prepared.prep(geom)
    # split_points = []
    # for spl in splitters:
        # if geom_checker.intersects(spl):
            # inters = geom.intersection(spl)
            # if inters.geom_type == 'Point':
                # split_points.append(inters)
            # else: # inters.geom_type == 'LineString':
                # print(inters.geom_type)
                # split_points.extend(inters.boundary.geoms)
    # return split_line_by_points(geom, split_points)


def split_line(feat: Dict[str, Any],
               endpoints: shapely.strtree.STRtree,
               lines: Dict[int, shapely.strtree.STRtree],
               layer_fld: Optional[str] = None,
               ) -> Generator[Dict[str, Any], None, None]:
    shape = shapely.geometry.shape(feat['geometry'])
    splitters = get_splitters(shape, endpoints, lines.get(feat['properties'].get(layer_fld, DEFAULT_LAYER)))
    # print(len(splitters))
    if splitters:
        if len(splitters) == 1:
            splits = shapely.ops.split(shape, splitters[0])
        else:
            splits = shapely.ops.split(shape, shapely.geometry.MultiPoint(splitters))
        for geom in splits:
            yield {
                'type': 'Feature',
                'geometry': shapely.geometry.mapping(geom),
                'properties': feat['properties']
            }
    else:
        yield feat


def get_splitters(shape: shapely.geometry.linestring.LineString,
                  endpoint_tree: shapely.strtree.STRtree,
                  line_tree: Optional[shapely.strtree.STRtree] = None,
                  ) -> Generator[Dict[str, Any], None, None]:
    shape_checker = shapely.prepared.prep(shape)
    shape_endpoints = shape.boundary.geoms
    splitters = endpoint_tree.query(shape)
    splitters = [
        pt for pt in splitters
        if pt not in shape_endpoints and shape_checker.intersects(pt)
    ]
    if line_tree:
        for line in line_tree.query(shape):
            if shape_checker.intersects(line):
                splitters.extend(
                    pt for pt in intersection_to_points(shape.intersection(line))
                    if pt not in shape_endpoints
                )
    return splitters


def intersection_to_points(geom) -> Iterable[shapely.geometry.point.Point]:
    if geom.geom_type == 'Point':
        yield geom
    elif geom.geom_type == 'MultiPoint':
        yield from geom
    elif geom.geom_type in ('LineString', 'MultiLineString'):
        yield from geom.boundary.geoms
    else:
        for subgeom in geom.geoms:
            yield from intersection_to_points(subgeom)


def create_connected_roads(source_file: os.PathLike,
                           target_file: os.PathLike,
                           layer_fld: Optional[str] = None,
                           cut_by_endpoints: bool = False,
                           ) -> None:
    print('loading splitters')
    endpoints, lines = read_splitters(
        source_file,
        layer_fld=layer_fld,
        endpoints_only=cut_by_endpoints
    )
    print('connecting roads')
    split_lines(source_file, endpoints, lines, layer_fld, target_file)


parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('in_roads',
    help='GDAL-compatible spatial file with roads'
)
parser.add_argument('out_roads',
    help='path to output a GDAL-compatible spatial file with connected roads'
)
parser.add_argument('-l', '--layer-fld', default='layer',
    help='vertical layer attribute name; omitted with -e'
)
parser.add_argument('-e', '--cut-by-endpoints', action='store_true',
    help='only cut by endpoints of other features'
)


if __name__ == '__main__':
    args = parser.parse_args()
    create_connected_roads(
        args.in_roads,
        args.out_roads,
        args.layer_fld,
        args.cut_by_endpoints,
    )