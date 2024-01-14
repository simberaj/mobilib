"""Eliminate exclaves (small noncontiguous parts of polygons).

Eliminates by merging exclaves with neighbouring polygons by longest shared
boundary.
"""

import operator
from typing import Tuple, List
from numbers import Number

import pandas as pd
import geopandas as gpd
import shapely.ops

import mobilib.argparser
import mobilib.core

AREA_GETTER = operator.attrgetter('area')


def strip_exclaves(geometry: gpd.GeoSeries) -> Tuple[gpd.GeoSeries, gpd.GeoDataFrame]:
    mains = []
    exclaves = []
    total_areas = []
    indices = []
    for index, geom in geometry.items():
        if geom.geom_type == 'Polygon':
            mains.append(geom)
        else:
            subgeoms = list(sorted(geom.geoms, key=AREA_GETTER))
            mains.append(subgeoms[-1])
            total_area = geom.area
            for exclave in subgeoms[:-1]:
                exclaves.append(exclave)
                total_areas.append(total_area)
                indices.append(index)
    exclaves = gpd.GeoDataFrame(
        {'parent_area': total_areas, 'parent_index': indices},
        crs=geometry.crs,
        geometry=exclaves,
    )
    exclaves['area'] = exclaves.geometry.map(AREA_GETTER)
    exclaves['parent_fraction'] = exclaves.eval('area / parent_area')
    return gpd.GeoSeries(mains), exclaves.drop('parent_area', axis=1)


def elimination_criterion(cols: List[str], max_size: float) -> str:
    if 'weight' in cols:
        size_col = 'weight'
    elif max_size < 1:
        size_col = 'parent_fraction'
    else:
        size_col = 'area'
    return f'{size_col} <= {max_size}'


def longest_boundary_neighbors(areas: gpd.GeoSeries,
                               neighbors: gpd.GeoSeries,
                               buffer: Number = 1,
                               ) -> pd.Series:
    return (
        gpd.sjoin(
            gpd.GeoDataFrame(geometry=areas.buffer(buffer)).rename_axis('index'),
            gpd.GeoDataFrame({'neighbor': neighbors}, geometry=neighbors)
        )
        .assign(boundary_area=lambda df: df.geometry.intersection(gpd.GeoSeries(df['neighbor'])).area)
        .reset_index()
        .set_index('index_right')
        .groupby('index')
        ['boundary_area'].idxmax()
    )


parser = mobilib.argparser.default(__doc__)
parser.add_argument(
    'area_file',
    help='polygon areas layer to eliminate exclaves in as a GDAL-compatible'
         ' file or semicolon-delimited CSV with WKT geometry'
)
parser.add_argument(
    'out_file',
    help='path to output file with areas with eliminated exclaves'
)
parser.add_argument(
    '-m', '--max-size',
    help='maximum exclave size to eliminate (measured by surface area by'
         ' default, if <1, regarded as fraction of total area size;'
         ' if not given, all exclaves will be eliminated)'
)
parser.add_argument(
    '-s', '--out-stats',
    help='output a file (GDAL spatial or CSV) with statistics about the'
         'exclaves to this path'
)
parser.add_argument(
    '-w', '--weighting-points',
    help='point layer to provide size attribute to exclaves (if weighting'
         ' column not given, points will be counted to give size) as a'
         ' GDAL-compatible file or semicolon-delimited CSV with x, y columns'
)
parser.add_argument(
    '-W', '--weighting-col',
    help='name of the weighting point layer attribute to give weight to each point'
)
parser.add_argument(
    '-x', '--x-col', default='X',
    help='name of the x-coordinate attribute in the weighting point layer (for CSV)'
)
parser.add_argument(
    '-y', '--y-col', default='Y',
    help='name of the y-coordinate attribute in the weighting point layer (for CSV)'
)
parser.add_argument(
    '-c', '--area-srid', default=3035,
    help='EPSG SRID of the area layer coordinates (for CSV)'
)
parser.add_argument(
    '-C', '--weighting-srid', default=3035,
    help='EPSG SRID of the weighting point layer coordinates (for CSV)'
)


if __name__ == '__main__':
    args = parser.parse_args()
    area_gdf = mobilib.core.read_gdf(args.area_file, srid=args.area_srid)
    area_gdf['geometry'], exclave_gdf = strip_exclaves(area_gdf['geometry'])
    exclave_gdf.reset_index(inplace=True)
    if args.weighting_points:
        weighting_gdf = mobilib.core.read_gdf(
            args.weighting_points, xcol=args.x_col, ycol=args.y_col, srid=args.weighting_srid
        )
        wcol = args.weighting_col
        if not wcol:
            wcol = '_point_count'
            weighting_gdf[wcol] = 1
        exclave_gdf['weight'] = gpd.sjoin(
            exclave_gdf,
            weighting_gdf[[wcol, 'geometry']],
            how='left', op='intersects'
        ).groupby('index')[wcol].sum()
    elim_crit_eq = elimination_criterion(exclave_gdf.columns.tolist(), args.max_size)
    exclave_gdf['eliminate'] = exclave_gdf.eval(elim_crit_eq)
    if args.out_stats:
        mobilib.core.write_gdf(exclave_gdf, args.out_stats)
    elim_target = pd.concat((
        longest_boundary_neighbors(
            exclave_gdf.loc[exclave_gdf['eliminate']].set_index('index').geometry,
            area_gdf.geometry,
        ),
        exclave_gdf[~exclave_gdf['eliminate']].set_index('index')['parent_index'],
    ))
    area_gdf['geometry'] = pd.concat((
        exclave_gdf[['geometry']].assign(target=elim_target),
        area_gdf['geometry'].rename_axis('target').reset_index(),
    )).groupby('target')['geometry'].apply(shapely.ops.unary_union)
    mobilib.core.write_gdf(area_gdf, args.out_file)
