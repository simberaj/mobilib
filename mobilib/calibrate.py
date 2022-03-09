'''Calibration to target sums (e.g. of MPD interaction data to municipal populations).'''

from typing import Tuple, Union

import numpy as np
import pandas as pd


def calibrate_source(value_df: pd.DataFrame,
                     calib_series: pd.Series,
                     ) -> pd.DataFrame:
    '''Calibration to a 1-D series.'''
    return _calibrate_by_index(value_df, calib_series, 0)


def calibrate_source_dirstat(value_df: pd.DataFrame,
                             calib_series: pd.Series,
                             ) -> pd.DataFrame:
    '''Calibration to outflow and inflow sums.'''
    if len(value_df.index.names) != 2 or len(calib_series.index.names) != 2:
        raise ValueError('need two ID fields for both datasets')
    if not frozenset(calib_series.index.levels[1]) == frozenset((True, False)):
        calib_id_frame = calib_series.index.to_frame()
        calib_id_frame.iloc[:,1] = (calib_id_frame.iloc[:,1] == calib_id_frame.iloc[:,0])
        calib_series = pd.Series(
            calib_series.values,
            index=pd.MultiIndex.from_frame(calib_id_frame)
        )
    value_calib = value_df.reset_index()
    value_calib['_is_self'] = value_calib.eval(' == '.join(value_df.index.names))
    value_calib.set_index([value_df.index.names[0], '_is_self'], inplace=True)
    to_name = value_df.index.names[1]
    calibrated_df = _calibrate_by_index(
        value_calib.drop(to_name, axis=1),
        calib_series,
        (0,1)
    ).reset_index().drop('_is_self', axis=1)
    calibrated_df[to_name] = value_calib[to_name].values
    return calibrated_df.set_index(value_df.index.names)


def _calibrate_by_index(value_df: pd.DataFrame,
                        calib_series: pd.Series,
                        levels: Union[int, Tuple[int]],
                        ) -> pd.DataFrame:
    if not hasattr(levels, '__len__'):
        levels = (levels, )
    value_factors = value_df.groupby(level=levels).sum()
    calib_sums = calib_series.groupby(level=levels).sum()
    # calib_fractions = value_sums.join(calib_sums, how='outer')
    for col in value_factors:
        value_factors[col] = (calib_sums / value_factors[col]).fillna(1)
    # raise RuntimeError
    with_coefs = value_df.join(
        value_factors,
        on=[name for i, name in enumerate(value_df.index.names) if i in levels],
        rsuffix='_coef'
    )
    for col in value_df.columns:
        with_coefs[col] *= with_coefs[col + '_coef']
    return with_coefs[value_df.columns.tolist()]



def calibrate_ipf(value_df: pd.DataFrame,
                  calib_series: pd.Series,
                  ) -> pd.DataFrame:
    '''Calibration by iterative proportional weighting.'''
    if len(value_df.index.names) != 2 or len(calib_series.index.names) != 2:
        raise ValueError('need two ID fields for both datasets')
    all_ids = np.array(list(sorted(
        frozenset(
            id for level in value_df.index.levels for id in level.unique()
        ) | frozenset(
            id for level in calib_series.index.levels for id in level.unique()
        )
    )))
    n_units = len(all_ids)
    value_indframe = value_df.index.to_frame()
    indexers = tuple(
        np.searchsorted(all_ids, value_indframe[col])
        for col in value_indframe.columns
    )
    marginals = [
        calib_series.groupby(level=level_i).sum()[all_ids].fillna(0).to_numpy()
        for level_i in (0, 1)
    ]
    out_df = pd.DataFrame({'_': 0}, index=pd.MultiIndex.from_product(
        (all_ids, all_ids), names=value_df.index.names
    ))
    for fld in value_df.columns:
        matrix = np.zeros((n_units, n_units), dtype=value_df[fld].dtype)
        matrix[indexers] = value_df[fld].fillna(0).to_numpy()
        matrix = mobilib.core.ipf(matrix, *marginals, max_iter=250)
        out_df[fld] = matrix.flatten()
    out_df.drop('_', axis=1, inplace=True)
    meaningful_rows = (out_df.to_numpy() != 0).any(axis=1)
    return out_df[meaningful_rows]


METHODS = {
    'source': calibrate_source,
    'dirstat': calibrate_source_dirstat,
    'ipf': calibrate_ipf,
}
