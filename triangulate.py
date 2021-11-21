import numpy as np
import scipy.spatial
import pandas as pd

import mobilib.argparser
import mobilib.core

parser = mobilib.argparser.default(__doc__, places=True)
parser.add_argument('out_file',
    help='path to output the CSV connecting points connected by triangulation edges'
)

if __name__ == '__main__':
    args = parser.parse_args()
    places = mobilib.core.read_places(args)
    xy = places[[args.x_col, args.y_col]].values
    delaunay = scipy.spatial.Delaunay(xy)
    indptr, tgt_indices = delaunay.vertex_neighbor_vertices
    # print((indptr, tgt_indices))
    src_index_flags = np.zeros_like(tgt_indices, dtype=bool)
    src_index_flags[indptr[:-1]] = True
    src_indices = src_index_flags.cumsum() - 1
    place_ids = places[args.place_id_col].values
    index_df = pd.DataFrame({
        f'from_{args.place_id_col}': place_ids[src_indices],
        f'to_{args.place_id_col}': place_ids[tgt_indices],
    })
    index_df.to_csv(args.out_file, sep=';', index=False)
    # print(xy)
    # areas.drop(['geometry'], axis=1).to_csv(args.out_file, sep=';', index=False)
