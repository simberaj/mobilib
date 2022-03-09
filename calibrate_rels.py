"""Calibrate interaction sizes to correct magnitude determined from other interaction data."""

import argparse

import pandas as pd

import mobilib.calibrate


METHOD_TYPE_LIST = list(mobilib.calibrate.METHODS.keys())


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
parser.add_argument('-o', '--omit-other', action='store_true',
    help='only output the calibrated columns and ID column(s) from the source file'
)
parser.add_argument('-S', '--no-suffix', action='store_true',
    help='do not add the calibration method suffix to the output calibrated columns'
    ' (warning: this might result in a duplicate column error if more methods are used)'
)


if __name__ == '__main__':
    args = parser.parse_args()
    calib_id_fields = args.calib_id_fields if args.calib_id_fields else args.id_fields
    calib_series = pd.read_csv(
        args.calib_file, sep=';'
    ).set_index(calib_id_fields)[args.calib_field]
    value_df = pd.read_csv(args.source_file, sep=';')
    if args.omit_other:
        base_df = value_df[args.id_fields]
    else:
        base_df = value_df.drop(args.value_field, axis=1)
    value_df_narrow = value_df[args.id_fields + args.value_field].set_index(args.id_fields)
    for method in args.type:
        out_df = mobilib.calibrate.METHODS[method](value_df_narrow, calib_series)
        if out_df is not None:
            if not args.no_suffix:
                out_df = out_df.rename(columns={
                    col: col + '_cal_' + method
                    for col in out_df.columns if col not in args.id_fields
                }),
            base_df = base_df.merge(
                out_df,
                left_on=args.id_fields,
                right_index=True,
                how='outer',
                suffixes=(False, False),
            )
    base_df.to_csv(args.target_file, index=False, sep=';')
