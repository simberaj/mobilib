import operator
from typing import List, Tuple, Optional, Dict, Union, Iterable

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely.ops
import shapely.geometry
import sklearn.cluster


import mobilib.neigh

AnyPolygon = Union[
    shapely.geometry.Polygon,
    shapely.geometry.MultiPolygon,
]


def equalize_polygons(polygons: gpd.GeoSeries,
                      subdivisions: Optional[gpd.GeoSeries] = None,
                      unsafe_geom: bool = False,
                      ) -> Tuple[gpd.GeoSeries, pd.DataFrame]:
    # target_area = polygons.area.quantile(.5)
    target_area = 20000000
    print('target area', target_area)
    # --- polygon aggregation
    agg_poly = gpd.GeoSeries(
        aggregate(polygons.tolist(), target_area),
        crs=polygons.crs,
    )
    # --- polygon splitting
    areas = agg_poly.area
    area_quots = (areas / target_area).round(0)
    subdiv_map = gpd.sjoin(
        gpd.GeoDataFrame(geometry=subdivisions.representative_point()),
        gpd.GeoDataFrame(geometry=agg_poly),
        how='inner',
        op='intersects'
    )['index_right']
    # return agg_poly, None
    ok_polygons = agg_poly[area_quots <= 1]
    poly_i = len(ok_polygons)
    out_id_map = list(zip(range(poly_i), ok_polygons.index))
    subdiv_polygons = []
    print(poly_i, end='     \r')
    for too_big_id, too_big_poly in agg_poly[area_quots > 1].items():
        agg_subdivs = aggregate(
            subdivisions[subdiv_map[subdiv_map == too_big_id].index].tolist(),
            target_area,
        )
        if unsafe_geom:
            agg_subdivs = match_geometry(agg_subdivs, too_big_poly)
        subdiv_polygons.extend(agg_subdivs)
        out_id_map.extend(zip(
            range(poly_i, poly_i + len(agg_subdivs)),
            [too_big_id] * len(agg_subdivs)
        ))
        poly_i += len(agg_subdivs)
        print(poly_i, end='     \r')
    return gpd.GeoSeries(pd.concat((
        ok_polygons,
        gpd.GeoSeries(subdiv_polygons, crs=ok_polygons.crs),
    ), ignore_index=True)), pd.DataFrame.from_records(out_id_map, columns=['id', 'orig_id'])


def match_geometry(subdivisions: List[AnyPolygon],
                   main_polygon: AnyPolygon,
                   ) -> List[AnyPolygon]:
    '''Match the subdivisions so that they fully subdivide the main polygon, without outreach.'''
    if np.isclose(main_polygon.area, sum(subdiv.area for subdiv in subdivisions)):
        return subdivisions
    main_polygon = main_polygon.buffer(0)
    cut_subdivs = [subdiv.intersection(main_polygon) for subdiv in subdivisions]
    remnant = main_polygon.difference(shapely.ops.unary_union(subdivisions))
    if remnant.is_empty:
        return cut_subdivs
    else:
        # return cut_subdivs + [remnant]
        # break the remnant down to individual components and eliminate them
        remnant_parts = remnant.geoms if hasattr(remnant, 'geoms') else [remnant]
        subdiv_comps = {}
        for remnant_part in remnant_parts:
            max_inters_len = -1
            to_merge_subdiv_i = 0
            for i, cut_subdiv in enumerate(cut_subdivs):
                inters_len = cut_subdiv.intersection(remnant_part).length
                if inters_len > max_inters_len:
                    max_inters_len = inters_len
                    to_merge_subdiv_i = i
            subdiv_comps.setdefault(to_merge_subdiv_i, []).append(remnant_part)
        return [
            shapely.ops.unary_union([cut_subdiv] + subdiv_comps.get(i, []))
            for i, cut_subdiv in enumerate(cut_subdivs)
        ]
        # remnant_triangles = [
            # remnant.intersection(triangle)
            # for triangle in shapely.ops.triangulate(remnant)
        # ]
        # return cut_subdivs + remnant_triangles


def aggregate(polygons: List[AnyPolygon],
              target_area: float,
              areas: Optional[List[float]] = None,
              neighbour_table: Optional[List[Tuple[int, int]]] = None,
              ) -> List[AnyPolygon]:
    if areas is None:
        areas = [poly.area for poly in polygons]
    max_addable_area = target_area * 1.5
    tot_weight = sum(a if a < max_addable_area else max_addable_area for a in areas)
    tgt_n = int(round(tot_weight / target_area))
    if len(polygons) <= tgt_n:
        return polygons
    elif tgt_n == 1:
        return shapely.ops.unary_union(polygons)
    else:
        group_labels = _centroid_kmeans([poly.centroid for poly in polygons], tgt_n, areas)
        # if neighbour_table is None:
            # neighbour_table = list(mobilib.neigh.neighbours(polygons))
        # neighbour_dict = _to_neighbour_dict(neighbour_table)
        # opt_labels = _optimize_aggregates(polygons, group_labels, neighbour_dict, target_area)
        opt_labels = group_labels
        return _union_groups(polygons, _create_groups(opt_labels))


def _to_neighbour_dict(neighbour_table: List[Tuple[int, int]]) -> Dict[int, List[int]]:
    neighbour_dict = {}
    for i1, i2 in neighbour_table:
        neighbour_dict.setdefault(i1, []).append(i2)
        neighbour_dict.setdefault(i2, []).append(i1)
    return neighbour_dict


def _centroid_kmeans(centroids: List[shapely.geometry.Point],
                     tgt_n: int,
                     weights: List[float],
                     ) -> List[List[int]]:
    coors = np.array([point.coords[0] for point in centroids])
    clusterer = sklearn.cluster.KMeans(n_clusters=tgt_n, random_state=1711)
    return clusterer.fit_predict(coors, sample_weight=np.array(weights)).tolist()


def _create_groups(group_labels: List[int]) -> List[List[int]]:
    groups = [[] for i in range(max(group_labels) + 1)]
    for i, label in enumerate(group_labels):
        groups[label].append(i)
    return groups


def _optimize_aggregates(polygons: List[AnyPolygon],
                         group_labels: List[int],
                         neighbour_dict: Dict[int, List[int]],
                         target_area: float,
                         ) -> List[int]:
    groups = _create_groups(group_labels)
    neighbour_edges = _neighbour_edge_lengths(polygons, neighbour_dict)
    aggregates = _union_groups(polygons, groups)
    candidates = list(sorted(_aggregate_updates(
        polygons, groups, group_labels, aggregates, neighbour_edges, target_area
    ), key=operator.itemgetter(1)))
    raise NotImplementedError
    # candidates.sort(key=operator.itemgetter(1))
    # while candidates:
        # # get the best change and perform it
        # best_change, best_crit = candidates.pop()
        # move_i, agg_from_i, agg_to_i = best_change
        # labels[move_i] = agg_to_i
        # aggregates[agg_from_i] = aggregates[agg_from_i].difference(polygons[move_i])
        # aggregates[agg_to_i] = aggregates[agg_to_i].union(polygons[move_i])
        # update = (agg_from_i, agg_to_i)
        # # remove all changes concerning these two aggregates
        # candidates = [
            # (change, crit) for change, crit in candidates
            # if change[1] not in update and change[2] not in update
        # ]
        # # add them back
        # for i1, i2 in neighbour_table:
    # TODO


def _aggregate_updates(polygons: List[AnyPolygon],
                       groups: List[List[int]],
                       group_labels: List[int],
                       aggregates: List[AnyPolygon],
                       neighbour_edges: Dict[int, Dict[int, float]],
                       target_area: float,
                       ) -> Iterable[Tuple[Tuple[int, int, int], float]]:
    # find all changes that are for the better
    for grp1_i, group1 in enumerate(groups):
        for poly1_i in group1:
            poly_cf = polygons[poly1_i].length
            poly_area = polygons[poly1_i].area
            inward_cf = 0
            outward_cfs = collections.defaultdict(float)
            for poly2_i, edge_len in neighbour_edges[poly1_i].items():
                if group_labels[poly2_i] == grp1_i:
                    inward_cf += edge_len
                else:
                    outward_cfs[group_labels[poly2_i]] += edge_len
            for grp2_i, outward_cf in outward_cfs.items():
                # criterion for moving poly1_i from grp1_i to grp2_i
                crit_delta = aggregation_criterion_delta(
                    area=poly_area,
                    cf=poly_cf,
                    inward_cf=inward_cf,
                    outward_cf=outward_cf,
                    from_agg=aggregates[grp1_i],
                    to_agg=aggregates[grp2_i],
                    target_area=target_area,
                )
                if crit_delta > 0:
                    yield ((poly1_i, grp1_i, grp2_i), crit_delta)


def aggregation_criterion(polygon: AnyPolygon, target_area: float) -> float:
    area = polygon.area
    return abs(1 - area / target_area) * polygon.length / (2 * math.sqrt(math.pi * area))


def aggregation_criterion_delta(area: float,
                                cf: float,
                                inward_cf: float,
                                outward_cf: float,
                                from_agg: AnyPolygon,
                                to_agg: AnyPolygon,
                                target_area: float,
                                ) -> float:
    from_area_after = from_agg.area - area
    to_area_after = to_agg.area + area
    return (
        (
            abs(1 - from_area_after / target_area)
            * (from_agg.length + 2 * inward_cf - cf)
            / (2 * math.sqrt(math.pi * from_area_after))
        )
        + (
            abs(1 - to_area_after / target_area)
            * (from_agg.length + cf - 2 * outward_cf)
            / (2 * math.sqrt(math.pi * to_area_after))
        )
        - aggregation_criterion(from_agg)
        - aggregation_criterion(to_agg)
    )


def _neighbour_edge_lengths(polygons: List[AnyPolygon],
                            neighbour_dict: Dict[int, List[int]],
                            ) -> Dict[int, Dict[int, float]]:
    return {
        i1: {i2: _shared_edge_length(polygons[i1], polygons[i2]) for i2 in neighs}
        for i1, neighs in neighbour_dict.items()
    }


def _shared_edge_length(poly1: AnyPolygon, poly2: AnyPolygon) -> float:
    inters = poly1.intersection(poly2)
    if inters.geom_type not in ('LineString', 'MultiLineString'):
        raise ValueError
    return inters.length


def _union_groups(polygons: List[AnyPolygon],
                  groups: List[List[int]],
                  ) -> List[AnyPolygon]:
    return [shapely.ops.unary_union([polygons[i] for i in group]) for group in groups]

