
'''Selects items from one CSV if their ids are present in columns of another CSV.'''

import argparse

import pandas as pd


GEN_ID_COL = 'hjz5f4h4ffh65'


parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('infile',
    help='input file as semicolon-delimited CSV'
)
parser.add_argument('selfile',
    help='selection column file as semicolon-delimited CSV'
)
parser.add_argument('outfile',
    help='path to output the selected CSV'
)
parser.add_argument('-i', '--idcol', default='id',
    help='name of the ID column in input file'
)
parser.add_argument('-s', '--selcol', nargs='+', default=['id'],
    help='name(s) of column(s) in selection file that contain input file ids to select'
)

if __name__ == '__main__':
    args = parser.parse_args()
    seldf = pd.read_csv(args.selfile, sep=';')
    sel_ids = set()
    for col in args.selcol:
        sel_ids.update(seldf[col])
    pd.read_csv(args.infile, sep=';').merge(
        pd.DataFrame({GEN_ID_COL : list(sel_ids)}),
        left_on=args.idcol,
        right_on=GEN_ID_COL,
    ).drop([GEN_ID_COL], axis=1).to_csv(args.outfile, sep=';', index=False)
