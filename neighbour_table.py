"""Generate table listing pairs of identifiers of neighbouring polygons."""

import pandas as pd

import mobilib.core
import mobilib.argparser
import mobilib.neigh


parser = mobilib.argparser.default(__doc__)
parser.add_argument('infile', help='input polygon table/GDAL-compatible file')
parser.add_argument('outfile', help='path to output neighbourhood table as CSV')
parser.add_argument('-i', '--id', default=None, help='area id column', metavar='COLNAME')


if __name__ == '__main__':
    args = parser.parse_args()
    data = mobilib.core.read_gdf(args.infile)
    geoms = data.geometry.tolist()
    ids = None if args.id is None else data[args.id]
    neighs = list(mobilib.neigh.neighbours(geoms, ids))
    df = pd.DataFrame.from_records(neighs, columns=['from_id', 'to_id'])
    df.to_csv(args.outfile, sep=';', index=False)
