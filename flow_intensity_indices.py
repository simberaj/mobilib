
import numpy as np
import pandas as pd

import mobilib.argparser


parser = mobilib.argparser.default(__doc__)
mobilib.argparser.add_interactions(parser, add_strength=False)

parser.add_argument('out_file',
    help='path to output the CSV with interactions and their relative strengths indicators'
)
parser.add_argument('-s', '--strength-col', nargs='+', default=['strength'],
    help='name of the flow strength attribute(s) in the interactions file'
)
parser.add_argument('-d', '--distance-col',
    help='name of the distance attribute in the interactions file'
)

if __name__ == '__main__':
    args = parser.parse_args()
    inter = pd.read_csv(args.inter_file, sep=';')
    from_id, to_id = args.from_id_col, args.to_id_col
    dist_col = args.distance_col
    inter_bidir = inter.query(f'{from_id} < {to_id}').merge(
        inter.query(f'{to_id} < {from_id}'),
        left_on=[from_id, to_id],
        right_on=[to_id, from_id],
        how='outer'
    )
    id_type = inter[to_id].dtype
    inter_bidir['id_one'] = inter_bidir[f'{from_id}_x'].fillna(
        inter_bidir[f'{to_id}_y']
    ).astype(id_type)
    inter_bidir['id_other'] = inter_bidir[f'{to_id}_x'].fillna(
        inter_bidir[f'{from_id}_y']
    ).astype(id_type)
    if dist_col:
        inter_bidir[dist_col] = inter_bidir[f'{dist_col}_x'].fillna(inter_bidir[f'{dist_col}_y'])
    for col in args.strength_col:
        inter_bidir[f'{col}_one'] = inter_bidir[f'{col}_x'].fillna(0)
        inter_bidir[f'{col}_other'] = inter_bidir[f'{col}_y'].fillna(0)
        inter_bidir[col] = inter_bidir.eval(f'{col}_one + {col}_other')
        inter_bidir[f'{col}_cpabs'] = 2 * np.where(
            inter_bidir[f'{col}_one'] < inter_bidir[f'{col}_other'],
            inter_bidir[f'{col}_one'],
            inter_bidir[f'{col}_other']
        )
        inter_bidir[f'{col}_cp'] = inter_bidir[f'{col}_cpabs'] / inter_bidir[col]
        if dist_col:
            inter_bidir[f'{col}_per_{dist_col}'] = inter_bidir.eval(f'{col} / {dist_col}')
            inter_bidir[f'{col}_cp_per_{dist_col}'] = inter_bidir.eval(f'{col}_cpabs / {dist_col}')
    inter_bidir.drop([
        col for col in inter_bidir.columns
        if col.endswith('_x') or col.endswith('_y')
    ], axis=1).to_csv(args.out_file, sep=';', index=False)
