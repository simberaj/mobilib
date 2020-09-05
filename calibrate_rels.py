import argparse
from typing import Tuple, Union

import numpy as np
import pandas as pd

import mobilib.relations


def calibrate_source(value_df: pd.DataFrame,
                     calib_series: pd.Series,
                     ) -> pd.DataFrame:
    return _calibrate_by_index(value_df, calib_series, 0)


def calibrate_source_dirstat(value_df: pd.DataFrame,
                             calib_series: pd.Series,
                             ) -> pd.DataFrame:
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
    value_sums = value_df.groupby(level=levels).sum()
    calib_sums = calib_series.groupby(level=levels).sum()
    # calib_fractions = value_sums.join(calib_sums, how='outer')
    for col in value_sums:
        value_sums[col] = calib_sums / value_sums[col]
    with_coefs = value_df.join(
        value_sums,
        on=[name for i, name in enumerate(value_df.index.names) if i in levels],
        rsuffix='_coef'
    )
    for col in value_df.columns:
        with_coefs[col] *= with_coefs[col + '_coef']
    return with_coefs[value_df.columns.tolist()]



def calibrate_ipf(value_df: pd.DataFrame,
                  calib_series: pd.Series,
                  ) -> pd.DataFrame:
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
        matrix = mobilib.relations.ipf(matrix, *marginals, max_iter=250)
        out_df[fld] = matrix.flatten()
    out_df.drop('_', axis=1, inplace=True)
    meaningful_rows = (out_df.to_numpy() != 0).any(axis=1)
    return out_df[meaningful_rows]


METHODS = {
    'source': calibrate_source,
    'dirstat': calibrate_source_dirstat,
    'ipf': calibrate_ipf,
}

METHOD_TYPE_LIST = list(METHODS.keys())


parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('calib_file',
    help='path to CSV with ground truth data to calibrate to'
)
parser.add_argument('calib_field',
    help='field in the calibration data to calibrate to'
)
parser.add_argument('source_file',
    help='CSV file with data to be calibrated'
)
parser.add_argument('target_file',
    help='path to output CSV file with calibrated data'
)
parser.add_argument('value_field', nargs='+',
    help='fields in the tested files to be calibrated'
)
parser.add_argument('-i', '--id-fields', nargs='+', default=['from_id', 'to_id'],
    help='ID column(s) in the source file to join it to the calibration file'
)
parser.add_argument('-c', '--calib-id-fields', nargs='+', default=[],
    help='ID column(s) in the calibration file to join it to the source file'
)
parser.add_argument('-t', '--type', nargs='+', default=METHOD_TYPE_LIST,
    help='calibration model type (' + ', '.join(METHOD_TYPE_LIST) + ')'
)


if __name__ == '__main__':
    args = parser.parse_args()
    calib_id_fields = args.calib_id_fields if args.calib_id_fields else args.id_fields
    calib_series = pd.read_csv(
        args.calib_file, sep=';'
    ).set_index(calib_id_fields)[args.calib_field]
    value_df = pd.read_csv(args.source_file, sep=';')
    base_df = value_df.drop(args.value_field, axis=1)
    value_df_narrow = value_df[args.id_fields + args.value_field].set_index(args.id_fields)
    for method in args.type:
        out_df = METHODS[method](value_df_narrow, calib_series)
        if out_df is not None:
            base_df = base_df.merge(
                out_df.rename(columns={
                    col: col + '_cal_' + method
                    for col in out_df.columns if col not in args.id_fields
                }),
                left_on=args.id_fields,
                right_index=True,
                how='outer',
                suffixes=(False, False),
            )
    base_df.to_csv(args.target_file, index=False, sep=';')
