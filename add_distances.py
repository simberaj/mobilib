
import numpy as np
import pandas as pd
import geopandas as gpd

import mobilib
import mobilib.argparser
import mobilib.routing

# TODO: input network (opt)

parser = mobilib.argparser.default(__doc__)
mobilib.argparser.add_interactions(parser, add_strength=False)
mobilib.argparser.add_places(parser)

parser.add_argument('out_file',
    help='path to output the CSV with interactions and their relative strengths'
)
parser.add_argument('-d', '--distance-col', default='distance',
    help='name of the distance attribute in the output file'
)
parser.add_argument('-D', '--distance-factor', type=float, default=1000,
    help='factor to divide distance with'
)
parser.add_argument('-n', '--network',
    help='compute network distances using this line network (a GDAL-compatible line file)'
)
parser.add_argument('-C', '--cost-attribute',
    help='use this network attribute as distance cost'
)


if __name__ == '__main__':
    args = parser.parse_args()
    inter = pd.read_csv(args.inter_file, sep=';')
    has_inter_geom = ('geometry' in inter.columns)
    if not has_inter_geom:
        # to always have place geometries in geometry_from and geometry_to
        inter['geometry'] = np.nan
    places = mobilib.read_places(args)
    inter_mg = inter.merge(
        places, left_on=args.from_id_col, right_on=args.place_id_col,
        how='left', suffixes=('', '_from')
    ).merge(
        places, left_on=args.to_id_col, right_on=args.place_id_col,
        how='left', suffixes=('', '_to')
    )
    unmatched_sources = inter_mg.loc[
        inter_mg['geometry_from'].isna(),args.from_id_col
    ].unique().tolist()
    if unmatched_sources:
        print(
            'source geometry not found for IDs',
            ', '.join(str(x) for x in unmatched_sources)
        )
    unmatched_targets = inter_mg.loc[
        inter_mg['geometry_to'].isna(),args.to_id_col
    ].unique().tolist()
    if unmatched_targets:
        print(
            'target geometry not found for IDs',
            ', '.join(str(x) for x in unmatched_targets)
        )
    if args.network is None:
        inter_mg[args.distance_col] = (
            gpd.GeoSeries(inter_mg['geometry_from']).distance(gpd.GeoSeries(inter_mg['geometry_to']))
            / args.distance_factor
        )
    else:
        geom_cols = ['geometry_from', 'geometry_to']
        print('reading network')
        network_gdf = gpd.read_file(args.network)
        all_points = pd.concat([inter_mg[col] for col in geom_cols]).unique()
        print('initializing finder')
        finder = mobilib.routing.RouteFinder(network_gdf)
        print('locating points')
        with finder.locate_points(all_points) as all_nodes:
            # node_lookup = {
                # pt.coords[0]: node
                # for pt, node in zip(all_points, all_nodes)
            # }
            print('computing paths')
            inter_mg[args.distance_col] = np.array([
                finder.cost(
                    # node_lookup[from_pt.coords[0]],
                    # node_lookup[to_pt.coords[0]],
                    from_pt.coords[0],
                    to_pt.coords[0],
                    args.cost_attribute,
                )
                for from_pt, to_pt
                    in inter_mg[geom_cols].itertuples(name=None, index=False)
            ]) / args.distance_factor
    out_flds = inter.columns.tolist() + [args.distance_col]
    if not has_inter_geom:
        out_flds.remove('geometry')
    inter_mg[out_flds].to_csv(args.out_file, sep=';', index=False)
