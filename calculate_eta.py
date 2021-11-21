# coding: utf8
"""Estimate the self-interaction parameter for interaction areal interpolation.

The self-interaction parameter is a correcting factor that gives more emphasis
on self-interactions when areally interpolating spatial interactions, thus
avoiding to create artificially much interaction where one unit is split into
many. This has not, in the end, proven itself to improve interpolation
accuracy much, but is left here for future reference.

For detailed description, refer to Šimbera, J., Aasa, A. (2019): Areal
interpolation of spatial interaction data. In: Gartner, G., Huang, H.:
Adjunct Proceedings of the 15th International Conference on Location Based
Services (DOI: 10.34726/lbs2019)

The script both prints a single parameter value for the whole system
(default) or fit a model for its estimation from input area values.
"""

import os
import argparse
import warnings
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd

import mobilib.neigh
import mobilib.argparser


def read_neigh_file(path: os.PathLike) -> List[Tuple[Any, Any]]:
    df = pd.read_csv(path, sep=';')
    return list(df.itertuples(index=False, name=None))


def calculate_etas(inter_df: pd.DataFrame, neigh: pd.DataFrame) -> pd.DataFrame:
    from_col, to_col, val_col = inter_df.columns[:3]
    flows_keyed = inter_df.set_index([from_col, to_col]).fillna(0).to_dict()[val_col]
    n_clipped = 0
    etas = []
    for fromid, toid in neigh:
        fromself = flows_keyed.get((fromid, fromid), 0)
        fromto = flows_keyed.get((fromid, toid), 0)
        tofrom = flows_keyed.get((toid, fromid), 0)
        toself = flows_keyed.get((toid, toid), 0)
        r = np.array([[fromself, fromto], [tofrom, toself]])
        tot = r.sum()
        tb = r.sum(axis=0) / tot
        etaparts = (r / r.sum(axis=1) - tb) / (np.diag(np.ones(2)) - tb)
        eta = (etaparts * r).sum() / tot
        if not (0 <= eta <= 1):
            n_clipped += 1
            eta = max(0, min(1, eta))
        etas.append((fromid, toid, eta, tot))
    if n_clipped:
        warnings.warn(f'{n_clipped} eta values outside 0-1 interval, clipping')
    return pd.DataFrame.from_records(etas, columns=[from_col, to_col, 'eta', 'weight'])


def show_global_mean(etadf):
    eta_mean = (etadf.eta * etadf.weight).sum() / etadf.weight.sum()
    eta_rmse = ((((etadf.eta - eta_mean) ** 2) * etadf.weight).sum() / etadf.weight.sum()) ** .5
    eta_mae = (abs(etadf.eta - eta_mean) * etadf.weight).sum() / etadf.weight.sum()
    print(f'Global mean {eta_mean:.6f} with RMSE {eta_rmse:.6f}, MAE {eta_mae:.6f}')


parser = mobilib.argparser.default(__doc__, interactions=True, areas=True)
parser.add_argument('eta_file',
    help='path to output the parameter value file as a semicolon-delimited CSV'
)
parser.add_argument('-n', '--neigh-file',
    help='interacting area neighbourhood file as a semicolon-delimited CSV'
)


if __name__ == '__main__':
    args = parser.parse_args()
    inter_df = pd.read_csv(args.inter_file, sep=';')
    area_gdf = gpd.read_file(args.area_file)
    if args.neigh_file:
        neigh = read_neigh_file(args.neigh_file)
    else:
        neigh = list(mobilib.neigh.neighbours(
            area_gdf.geometry.tolist(),
            area_gdf.area_id.tolist(),
        ))
    eta_df = calculate_etas(
        inter_df[[args.from_id_field, args.to_id_field, args.strength_field]],
        neigh
    )
    show_global_mean(eta_df)
    eta_df.to_csv(args.eta_file, sep=';', index=False)
