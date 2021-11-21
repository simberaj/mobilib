"""Utilities for route finding on geographical networks."""

import contextlib
from typing import Any, Dict, Optional, List, Union, Tuple, Iterable

import numpy as np
import geopandas as gpd
import networkx as nx
import shapely.geometry
import shapely.strtree


class RouteFinder:
    def __init__(self,
                 line_gdf: gpd.GeoDataFrame,
                 attrs: List[str] = [],
                 tolerance: float = .001,
                 max_match_distance: float = np.inf,
                 base_line_search_distance: float = 25.,
                 base_point_search_distance: float = .1,
                 ):
        self.network = from_lines(line_gdf, attrs=attrs, tolerance=tolerance)
        self.line_tree = nearest_search_tree(
            line_gdf.geometry,
            base_search_distance=base_line_search_distance,
        )
        self.node_tree = nearest_search_tree(
            [shapely.geometry.Point(*pt) for pt in self.network.nodes()],
            base_search_distance=base_point_search_distance,
        )
        self.max_match_distance = max_match_distance
    
    def cost(self,
             from_node: Tuple[float, float],
             to_node: Tuple[float, float],
             attr: str,
             ) -> Union[None, float]:
        try:
            return nx.shortest_path_length(
                self.network,
                from_node, to_node,
                weight=attr
            )
        except nx.NetworkXNoPath:
            return None

    @contextlib.contextmanager
    def locate_points(self,
                      points: Iterable[shapely.geometry.point.Point],
                      ) -> List[Tuple[float, float]]:
        nodes = []
        added_edges = []
        for point in points:
            match_line = self.line_tree.nearest(point)
            distance = point.distance(match_line)
            if distance > self.max_match_distance:
                nodes.append(None)
            else:
                # get nodes to which this line is connected to
                added_node = tuple(point.coords[0])
                from_node, to_node = [
                    tuple(self.node_tree.nearest(pt).coords[0])
                    for pt in match_line.boundary.geoms
                ]
                nodes.append(added_node)
                added_edges.extend(self._joining_edges(from_node, added_node, to_node))
                if self.network.has_edge(to_node, from_node):
                    added_edges.extend(self._joining_edges(to_node, added_node, from_node))
        self.network.add_edges_from(added_edges)
        yield nodes
        self.network.remove_edges_from(added_edges)
    
    def _joining_edges(self, from_node, added_node, to_node):
        from_pt, added_pt, to_pt = [
            shapely.geometry.Point(node)
            for node in (from_node, added_node, to_node)
        ]
        from_dist, to_dist = added_pt.distance(from_pt), added_pt.distance(to_pt)
        total_dist = from_dist + to_dist
        edge = self.network[from_node][to_node]
        from_attrs = self.part_edge(edge, from_dist / total_dist)
        from_attrs['geometry'] = shapely.geometry.LineString([from_pt, added_pt])
        to_attrs = self.part_edge(edge, to_dist / total_dist)
        to_attrs['geometry'] = shapely.geometry.LineString([added_pt, to_pt])
        return [
            (from_node, added_node, from_attrs),
            (added_node, to_node, to_attrs),
        ]
    
    @staticmethod
    def part_edge(edge, coef):
        return {
            key: (val * coef if isinstance(val, (int, float)) else val)
            for key, val in edge.items()
        }
    
    def get_nearest_line(self,
                         point: shapely.geometry.point.Point,
                         ) -> Union[shapely.geometry.linestring.LineString, None]:
        nearest = self.line_tree.nearest(point)
        distance = point.distance(nearest)
        return None if distance > self.max_match_distance else nearest
            

class PatchedSTRtree:
    """An upgrade of shapely's STR tree to allow efficient 'nearest' queries."""
    def __init__(self,
                 geoms: List[shapely.geometry.base.BaseGeometry],
                 base_search_distance: float = 1.,
                 expand_coef: float = np.e,
                 max_search_distance: float = np.inf,
                 ):
        self._tree = shapely.strtree.STRtree(geoms)
        self.base_search_distance = base_search_distance
        self.max_search_distance = max_search_distance
        self.expand_coef = expand_coef
    
    def query(self, geom):
        return self._tree.query(geom)
    
    def nearest(self, geom):
        dist = self.base_search_distance
        geoms_in_dist = self._tree.query(geom.buffer(dist))
        while not geoms_in_dist and dist < self.max_search_distance:
            dist *= self.expand_coef
            geoms_in_dist = self._tree.query(geom.buffer(dist))
        return min(geoms_in_dist, key=geom.distance)


def nearest_search_tree(geoms: List[shapely.geometry.base.BaseGeometry], **kwargs):
    if hasattr(shapely.strtree.STRtree, 'nearest'):
        # newer version of shapely has this strtree functionality
        return shapely.strtree.STRtree(geoms)
    else:
        return PatchedSTRtree(geoms, **kwargs)


def reversed_edge_attrs(attrs: Dict[str, Any]) -> Dict[str, Any]:
    attrs = attrs.copy()
    attrs['geometry'] = shapely.geometry.LineString(
        list(attrs['geometry'].coords)[::-1]
    )
    return attrs


def from_lines(gdf: gpd.GeoDataFrame,
               attrs: List[str] = [],
               tolerance: float = .001,
               ) -> nx.DiGraph:
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
    )


def to_lines(graph: nx.DiGraph,
             crsdef: Dict[str, Any],
             ) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        nx.to_pandas_edgelist(graph).drop(['source', 'target'], axis=1),
        crs=crsdef
    )


def fix_orientation(graph: nx.DiGraph,
                    oneway_col: Optional[str] = None,
                    forward_way_value: Optional[Any] = None,
                    backward_way_value: Optional[Any] = None,
                    ) -> None:
    if oneway_col:
        apply_oneway_attr(graph, oneway_col, forward_way_value, backward_way_value)
    else:
        bidirectionalize(graph)


def apply_oneway_attr(graph: nx.DiGraph,
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


def largest_component(graph: nx.DiGraph) -> nx.DiGraph:
    return graph.subgraph(max(
        nx.weakly_connected_components(graph), key=len
    ))

