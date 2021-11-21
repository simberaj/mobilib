
import pandas as pd
import shapely.geometry

import mobilib
import mobilib.argparser
import mobilib.core
import mobilib.relations


parser = mobilib.argparser.default(
    __doc__, places=True, interactions=True, add_interaction_strength=False
)
parser.add_argument('out_file',
    help='path to output the line CSV'
)

if __name__ == '__main__':
    args = parser.parse_args()
    inter_df = pd.read_csv(args.inter_file, sep=';')
    orig_cols = inter_df.columns.tolist()
    id_cols = [args.from_id_col, args.to_id_col]
    rem_cols = [col for col in orig_cols if col not in id_cols]
    place_df = mobilib.core.read_places(args)
    line_df = mobilib.relations.to_lines(
        inter_df[id_cols + rem_cols],
        place_df.set_index(args.place_id_col),
    )
    line_df = line_df[orig_cols + line_df.columns.tolist()[len(orig_cols):]]
    line_df.to_csv(args.out_file, sep=';', index=False)
