
import pandas as pd
import shapely.geometry

import mobilib
import mobilib.argparser


parser = mobilib.argparser.default(
    __doc__, places=True, interactions=True, add_interaction_strength=False
)
parser.add_argument('out_file',
    help='path to output the line CSV'
)

if __name__ == '__main__':
    args = parser.parse_args()
    inter_df = pd.read_csv(args.inter_file, sep=';')
    place_df = mobilib.read_places(args)
    all_df = inter_df.merge(
        place_df.rename(columns=lambda colname: 'from_' + str(colname)),
        left_on=args.from_id_col,
        right_on='from_' + args.place_id_col,
    )
    all_df = all_df.merge(
        place_df.rename(columns=lambda colname: 'to_' + str(colname)),
        left_on=args.to_id_col,
        right_on='to_' + args.place_id_col,
    )
    all_df['geometry'] = [
        shapely.geometry.LineString([pt1, pt2]).wkt
        for pt1, pt2 in zip(all_df['from_geometry'], all_df['to_geometry'])
    ]
    all_df.drop([
        'from_geometry', 'to_geometry',
        'from_' + args.place_id_col, 'to_' + args.place_id_col
    ], axis=1).to_csv(args.out_file, sep=';', index=False)