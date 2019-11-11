
import numpy as np
import pandas as pd
import geopandas as gpd

import mobilib
import mobilib.argparser

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
    inter_mg[args.distance_col] = (
        gpd.GeoSeries(inter_mg['geometry_from']).distance(gpd.GeoSeries(inter_mg['geometry_to']))
        / args.distance_factor
    )
    out_flds = inter.columns.tolist() + [args.distance_col]
    if not has_inter_geom:
        out_flds.remove('geometry')
    inter_mg[out_flds].to_csv(args.out_file, sep=';', index=False)
