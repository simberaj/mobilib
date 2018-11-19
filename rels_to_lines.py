import argparse

import pandas as pd
import shapely.geometry

def makeline(x1, y1, x2, y2):
    pt1 = shapely.geometry.Point(x1, y1)
    pt2 = shapely.geometry.Point(x2, y2)
    geom = shapely.geometry.LineString([pt1, pt2])
    return geom.wkt


parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('relfile',
    help='input relations as semicolon-delimited CSV'
)
parser.add_argument('sitefile',
    help='input site locations as semicolon-delimited CSV'
)
parser.add_argument('linefile',
    help='path to output the line CSV'
)
parser.add_argument('-n', '--name', default='site_id',
    help='site id column in site file'
)
parser.add_argument('-x', '--xcol', default='x',
    help='name of the X-coordinate column in site file'
)
parser.add_argument('-y', '--ycol', default='y',
    help='name of the Y-coordinate column in site file'
)
parser.add_argument('-f', '--from-id', 
    help='name of the from column in relation file'
)
parser.add_argument('-t', '--to-id',
    help='name of the to column in relation file'
)

if __name__ == '__main__':
    args = parser.parse_args()
    reldf = pd.read_csv(args.relfile, sep=';')
    sitedf = pd.read_csv(args.sitefile, sep=';')
    from_id = args.from_id if args.from_id else 'from_' + args.name
    to_id = args.to_id if args.to_id else 'to_' + args.name
    # print(sitedf.rename(lambda colname: str(colname) + '_from').dtypes)
    alldf = reldf.merge(
        sitedf.rename(columns=lambda colname: 'from_' + str(colname)),
        left_on=from_id,
        right_on='from_' + args.name,
    ).merge(
        sitedf.rename(columns=lambda colname: 'to_' + str(colname)),
        left_on=to_id,
        right_on='to_' + args.name,
    )
    alldf['line'] = [makeline(*vals) for vals in zip(
        alldf['from_' + args.xcol],
        alldf['from_' + args.ycol],
        alldf['to_' + args.xcol],
        alldf['to_' + args.ycol],
    )]
    alldf.to_csv(args.linefile, sep=';', index=False)
    # print(alldf.head())
    # print(alldf.dtypes)