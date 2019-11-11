# coding: utf8

import shapely
import shapely.wkt
import shapely.ops
import shapely.strtree
import shapely.prepared


def neighbours(geoms, gids=None):
    if gids is None:
        gids = list(range(len(geoms)))
    mem_to_ids = {id(geom) : gid for geom, gid in zip(geoms, gids)}
    print('found', len(geoms), 'geometries')
    geomtree = shapely.strtree.STRtree(geoms)
    print('str tree built')
    for gid, geom in zip(gids, geoms):
        prepgeom = shapely.prepared.prep(geom)
        print(gid, end=(' ' * 20 + '\r'))
        for neigh_geom in geomtree.query(geom):
            if prepgeom.intersects(neigh_geom):
                neigh_gid = mem_to_ids[id(neigh_geom)]
                if neigh_gid > gid:
                    yield (gid, neigh_gid)


def fix_polygons(geoms, tolerance=.01):
    geoms = geoms.copy()
    for from_i, to_i in neighbours(geoms.copy()):
        if from_i > to_i:
            from_i, to_i = to_i, from_i
        geoms.iloc[from_i] = shapely.ops.snap(
            geoms.iloc[from_i],
            geoms.iloc[to_i],
            tolerance,
        )
    return geoms



    
if __name__ == '__main__':
    n = 4
    tests = [
        shapely.wkt.loads('POLYGON(({a} {b}, {c} {b}, {c} {d}, {a} {d}, {a} {b}))'.format(
            a=i, b=j, c=i+1, d=j+1
        ))
        for i in range(n)
        for j in range(n)
    ]
    # for id, t in tests: print(id, t)
    # print(list(neighbours(list(range(len(tests))), tests)))
    print(list(neighbours(tests)))