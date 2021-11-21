"""Compute weighted area centroids based on overlay with another layer.

The centroids will be determined as centroids of the intersection of the
weighting layer with each area.
"""

from typing import Iterable, Collection

import geopandas as gpd
import shapely.ops
import shapely.strtree
import shapely.geometry

import mobilib.argparser


def weighted_centroids(areas: Iterable[shapely.geometry.base.BaseGeometry],
                       weighters: Collection[shapely.geometry.base.BaseGeometry],
                       labels: Iterable[str]
                       ) -> Iterable[shapely.geometry.base.BaseGeometry]:
    weighters_tree = shapely.strtree.STRtree(weighters)
    for geom, lbl in zip(areas, labels):
        if not geom.is_valid:
            geom = geom.buffer(0)
        weighted = shapely.ops.unary_union([
            weighter.intersection(geom)
            for weighter in weighters_tree.query(geom)
            if weighter.intersects(geom)
        ])
        if weighted.is_empty:
            yield geom.centroid
        else:
            yield weighted.centroid


parser = mobilib.argparser.default(__doc__)
parser.add_argument(
    'area_file',
    help='path to the GDAL-compatible file with the polygon areas to calculate centroids for'
)
parser.add_argument(
    'weight_file',
    help='path to the GDAL-compatible file with the polygon overlay weighting layer'
)
parser.add_argument(
    'output_file',
    help='path to output the weighted centroids file as a semicolon-delimited GeoCSV'
)
parser.add_argument(
    '-s', '--output-crs',
    help='EPSG ID of the CRS to compute the centroids in (default: use area CRS)'
)
parser.add_argument(
    '-C', '--input-encoding', default='utf-8',
    help='text encoding of the attribute table of the area file'
)


if __name__ == '__main__':
    args = parser.parse_args()
    area_gdf = gpd.read_file(args.area_file)
    if args.output_crs is not None:
        area_gdf = area_gdf.to_crs(args.output_crs, encoding=args.input_encoding)
    target_crs = area_gdf.crs
    weight_gdf = gpd.read_file(args.weight_file).to_crs(target_crs)
    centroids = list(weighted_centroids(
        area_gdf.geometry.values,
        weight_gdf.geometry.values,
        area_gdf['name'].values,
    ))
    out_gdf = area_gdf.copy()
    out_gdf.geometry = centroids
    out_gdf.to_file(args.output_file, encoding='utf-8')
