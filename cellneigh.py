import sys
import operator
import itertools

import numpy
import pandas as pd
import geopandas as gpd
import shapely.geometry
import shapely.ops
import shapely.prepared
import scipy.spatial


import geohash

LOC_COLS = ['loc_x', 'loc_y']
NEIGH_NAMES = ['neigh1', 'neigh2']
# LOC_DECIMALS = 5

ZERO_POINT = shapely.geometry.Point(0.0, 0.0)

def to_gdf(df):
    # for col in LOC_COLS:
        # df[col] = df[col].round(LOC_DECIMALS)
    pts = [
        shapely.geometry.Point(xy) 
            if not numpy.isnan(xy).sum() else shapely.geometry.Point()
        for xy in zip(*(df[col] for col in LOC_COLS))
    ]
    return gpd.GeoDataFrame(
        df.drop(LOC_COLS, axis=1),
        crs={'init': 'epsg:4326'},
        geometry=pts
    )
    
def multipoint(pts):
    return shapely.geometry.MultiPoint([
        pt for pt in pts
        if not (pt.is_empty or pt == ZERO_POINT)
    ])
    
def point_centroid(pts):
    return multipoint(pts).centroid
    
def point_spread(pts):
    bounds = multipoint(pts).bounds
    if bounds:
        return (bounds[3] - bounds[1]) * (bounds[2] - bounds[0])
    else:
        return 0
        
def flatten_index(df):
    df.columns = ['_'.join(col).strip('_') for col in df.columns.values]
    return df
    
def geohash_encode(pt):
    return geohash.encode(pt.y, pt.x, 9) if not pt.is_empty else None
    
def pointset_bounds(coords):
    return (
        min(coords, key=operator.itemgetter(0))[0],
        min(coords, key=operator.itemgetter(1))[1],
        max(coords, key=operator.itemgetter(0))[0],
        max(coords, key=operator.itemgetter(1))[1],
    )
    
def voronoi_neighbours(ids, points, extent=None):
    pt_coords = [(pt.x, pt.y) for pt in points]
    pt_coords.extend(bounds_to_limiting_generators(*pointset_bounds(pt_coords)))
    diagram = scipy.spatial.Voronoi(pt_coords)
    vertex_regions = {}
    # regions_for_points = list(diagram.point_region)
    # print(diagram.regions)
    # print(regions_for_points)
    if extent is not None:
        extent = shapely.prepared.prep(extent)
    for pt_i, reg_i in enumerate(diagram.point_region[:-4]):
        for vert_i in diagram.regions[reg_i]:
            if vert_i >= 0:
                vertex_regions.setdefault(vert_i, []).append(ids[pt_i])
    neighs = set()
    for vert_i, idlist in vertex_regions.items():
        if not extent or extent.contains(shapely.geometry.Point(*diagram.vertices[vert_i])):
            neighs.update(
                tuple(sorted(pair))
                for pair in itertools.combinations(idlist, 2)
            )
    return sorted(neighs)
    
    # lines = [
        # shapely.geometry.LineString(diagram.vertices[line])
        # for line in diagram.ridge_vertices
        # if -1 not in line
    # ]
    # df = gpd.GeoDataFrame(index=ids, geometry=points, crs={'init' : 'epsg:25833'})
    # cells = gpd.GeoDataFrame(
        # geometry=gpd.GeoSeries(
            # intersect_with_extent(shapely.ops.polygonize(lines), extent),
            # crs={'init' : 'epsg:25833'}
        # ),
        # crs={'init' : 'epsg:25833'},
    # )
    # return gpd.sjoin(cells, df, how='inner', op='intersects').set_index('index_right').geometry
    # # joined.set_index(
    # # print(joined.head())
    # # print()
    # # print('FIN')
    # # print()
    # # return joined.geometry
    
    
# def intersect_with_extent(polygons, extent):
    # extent_prep = shapely.prepared.prep(extent)
    # for polygon in polygons:
        # if extent_prep.contains_properly(polygon):
            # yield polygon
        # else:
            # polygon = extent.intersection(polygon)
            # if polygon and not polygon.is_empty:
                # yield polygon
    
        
    
def bounds_to_limiting_generators(minx, miny, maxx, maxy):
    addx = maxx - minx
    addy = maxy - miny
    return [
        (minx - addx, miny - addy),
        (maxx + addx, miny - addy),
        (minx - addx, maxy + addy),
        (maxx + addx, maxy + addy),
    ]
    
def broadcast_site_ids(site_neighs, cids_per_site):
    for neigh1, neigh2 in site_neighs:
        cids1 = cids_per_site[neigh1]
        cids2 = cids_per_site[neigh2]
        yield from itertools.product(cids1, cids2)
    for site_cidlist in cids_per_site:
        yield from itertools.combinations(site_cidlist, 2)
    

if __name__ == '__main__':
    df = pd.read_csv(sys.argv[1], sep=';')
    gdf = to_gdf(df)
    gdf['geohash'] = gdf.geometry.apply(geohash_encode)
    gdf = gdf.to_crs(epsg=25833)
    # spl = gdf.dsc.str.split(expand=True, n=1)
    # gdf['name'], gdf['code'] = spl[0], spl[1]
    # print(gdf.head())
    # print(sites.head())
    sitegdf = flatten_index(gdf.groupby('geohash').agg({
        'geometry' : [multipoint, point_centroid, point_spread],
        'cid' : list,
    }))
    
    countrydf = gpd.read_file(sys.argv[2])
    country_shape = countrydf.loc[countrydf.index[0],'geometry']
    
    siteneighs = voronoi_neighbours(
        ids=sitegdf.index,
        points=sitegdf.geometry_point_centroid,
        extent=country_shape
    )
    
    # cids = pd.Series(gdf.cid, index=gdf.geohash)
    # print(siteneighs[:10])
    # print(sitegdf.cid_list)
    
    pd.DataFrame.from_records(
        list(broadcast_site_ids(siteneighs, sitegdf.cid_list)),
        columns=NEIGH_NAMES
    ).sort_values(NEIGH_NAMES).to_csv(sys.argv[3], sep=';', index=False)
    # site_voronoi = voronoi_cells(sitegdf.index, sitegdf.geometry_point_centroid, country_shape)
    # gdf = 
    # print(site_voronoi)
    # # print(country_shape)
    # pts = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
    # ids = ['pt' + str(i) for i in range(len(pts))]
    # shapely_pts = [shapely.geometry.Point(*pt) for pt in pts]
    # neighs = voronoi_neighbours(ids, shapely_pts)
    # for i in range(len(ids)):
        # print(ids[i], pts[i]) #, neighs[i])
    # for n in neighs: print(n)
    