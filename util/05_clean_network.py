'''Connect (cut) roads so that they only touch at endpoints.'''

import os
import json
import shutil
import argparse
import tempfile
import operator
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import shapely.geometry


def reversed_edge_attrs(attrs: Dict[str, Any]) -> Dict[str, Any]:
    attrs = attrs.copy()
    attrs['geometry'] = shapely.geometry.LineString(
        list(attrs['geometry'].coords)[::-1]
    )
    return attrs


def correct_oneway_edges(graph: nx.DiGraph,
                         attribute: str,
                         forward_value: Optional[str] = None,
                         backward_value: Optional[str] = None,
                         ) -> None:
    to_add = []
    to_remove = []
    for node1, node2, eattrs in graph.out_edges(data=True):
        oneway = eattrs.get(attribute, None)
        del eattrs[attribute]
        if oneway == forward_value and forward_value is not None:
            pass # everything is as expected
        elif oneway == backward_value and backward_value is not None:
            # we need to reverse edge direction
            if not graph.has_edge(node2, node1):
                to_add.append((node2, node1, reversed_edge_attrs(eattrs)))
            to_remove.append((node1, node2))
        else:
            # add edge in the opposite direction, too
            if not graph.has_edge(node2, node1):
                to_add.append((node2, node1, reversed_edge_attrs(eattrs)))
    graph.remove_edges_from(to_remove)
    graph.add_edges_from(to_add)


def bidirectionalize(graph: nx.DiGraph) -> None:
    graph.add_edges_from([
        (node2, node1, reversed_edge_attrs(eattrs))
        for node1, node2, eattrs in graph.out_edges(data=True)
        if not graph.has_edge(node2, node1)
    ])

def roads_to_network(source_file: os.PathLike,
                     target_file: os.PathLike,
                     attrs: List[str] = [],
                     oneway_col: Optional[str] = None,
                     forward_way_value: Optional[Any] = None,
                     backward_way_value: Optional[Any] = None,
                     tolerance: float = 0.,
                     ) -> None:
    if attrs and oneway_col not in attrs:
        attrs.append(oneway_col)
    network, crsdef = load_network_graph(source_file, attrs, tolerance)
    if oneway_col:
        correct_oneway_edges(network, oneway_col, forward_way_value, backward_way_value)
    else:
        bidirectionalize(network)
    network = network.subgraph(max(
        nx.weakly_connected_components(network), key=len
    ))
    save_network_graph(network, target_file, crsdef)


def load_network_graph(source_file: os.PathLike,
                       attrs: List[str] = [],
                       tolerance: float = 0.,
                       ) -> Tuple[nx.DiGraph, Dict[str, Any]]:
    gdf = gpd.read_file(source_file)
    if attrs:
        gdf.drop(
            [col for col in gdf.columns if col not in attrs and col != 'geometry'],
            inplace=True, axis=1,
        )
    gdf['_source'] = gdf.geometry.map(lambda geom: geom.coords[0])
    gdf['_target'] = gdf.geometry.map(lambda geom: geom.coords[-1])
    if tolerance > 0:
        rounding = int(np.ceil(-np.log10(tolerance)))
        def rounder(tup):
            return tuple(round(value, rounding) for value in tup)
        gdf['_source'] = gdf['_source'].map(rounder)
        gdf['_target'] = gdf['_target'].map(rounder)
    return nx.from_pandas_edgelist(
        gdf, '_source', '_target', edge_attr=True, create_using=nx.DiGraph
    ), gdf.crs

# def save_network_graph(graph: nx.DiGraph,
                       # source_file: os.PathLike,
                       # target_file: os.PathLike,
                       # ) -> None:
    # source_basepath, _ = os.path.splitext(source_file)
    # target_basepath, _ = os.path.splitext(target_file)
    # with tempfile.TemporaryDirectory() as dirpath:
        # nx.write_shp(graph, dirpath)
        # temp_basepath = os.path.join(dirpath, 'edges')
        # for ext in ('shp', 'shx', 'dbf'):
            # shutil.copyfile(temp_basepath + '.' + ext, target_basepath + '.' + ext)
    # for ext in ('prj', ):
        # # TODO dunno how nx handles encodings
        # source_path = source_basepath + '.' + ext
        # if os.path.isfile(source_path):
            # shutil.copyfile(source_path, target_basepath + '.' + ext)

def save_network_graph(graph: nx.DiGraph,
                       target_file: os.PathLike,
                       crsdef: Dict[str, Any],
                       ) -> None:
    gdf = gpd.GeoDataFrame(
        nx.to_pandas_edgelist(graph).drop(['source', 'target'], axis=1),
        crs=crsdef
    )
    gdf.to_file(target_file)
    # # df['from_i'] = df['_source']
    # # nodes = list(graph.nodes())
    # print(gdf.dtypes)
    # print(gdf.head())
    # # print(nodes[:10])
    # # print(crsdef)
    # raise RuntimeError


parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('in_roads',
    help='GDAL-compatible spatial file with connected (routable) roads'
)
parser.add_argument('out_network',
    help='path to output a GDAL-compatible spatial file with cleaned roads of the main network'
)
parser.add_argument('-a', '--attrs', nargs='+',
    help='attributes of the roads to preserve in the network (default: preserve all)'
)
parser.add_argument('-o', '--oneway-col', default='oneway',
    help='oneway street attribute name, will be used to determine edge directions and dropped'
)
parser.add_argument('-f', '--forward-way-value',
    help='oneway street attribute value signifying rideability only in the direction of geometry (F in OSM)'
)
parser.add_argument('-r', '--reverse-way-value',
    help='oneway street attribute value signifying rideability against the direction of geometry (T in OSM)'
)
parser.add_argument('-t', '--tolerance', type=float, default=0.001,
    help='geometry tolerance for node snapping'
)


if __name__ == '__main__':
    args = parser.parse_args()
    roads_to_network(
        args.in_roads,
        args.out_network,
        args.attrs,
        args.oneway_col,
        args.forward_way_value,
        args.reverse_way_value,
        args.tolerance,
    )