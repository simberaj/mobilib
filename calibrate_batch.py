import os
import argparse

import numpy as np
import pandas as pd
import sklearn.linear_model

import mobilib.relations


class LADRegression:
    def __init__(self, epsilon=1e-9, delta=1e-3, **kwargs):
        self.epsilon = epsilon
        self.delta = delta
        self.inner = sklearn.linear_model.LinearRegression(**kwargs)

    def fit(self, X, y, sample_weights=None):
        if sample_weights is None:
            sample_weights = np.ones(y.shape[0])
        prev_tae = np.inf
        tgtsum = y.sum()
        while True:
            self.inner.fit(X, y, sample_weights)
            fitted = self.inner.predict(X)
            resid = abs(fitted - y)
            tae = resid.sum() / tgtsum
            if abs(tae - prev_tae) < self.epsilon:
                break
            prev_tae = tae
            sample_weights = 1 / np.where(resid < self.delta, self.delta, resid)
        self.coef_ = self.inner.coef_
        self.intercept_ = self.inner.intercept_

    def predict(self, X):
        return self.inner.predict(X)


class SumMatchingMultiplication:
    def fit(self, X, y):
        self.multiplier = y.sum() / X.sum()
        self.coef_ = [self.multiplier]
        self.intercept_ = 0.

    def predict(self, X):
        return (X * self.multiplier).sum(axis=1)
        

class LogRegression:
    def __init__(self, inner):
        self.inner = inner
    
    def fit(self, X, y, sample_weights=None):
        self.inner.fit(np.log10(X), np.log10(y), sample_weights)
    
    def predict(self, X):
        return 10 ** self.inner.predict(np.log10(X))
    
    @property
    def coef_(self):
        return self.inner.coef_

    @property
    def intercept_(self):
        return self.inner.intercept_


MODEL_TYPES = {
    'lad' : LADRegression(fit_intercept=False),
    'ols' : sklearn.linear_model.LinearRegression(fit_intercept=False),
    'sum' : SumMatchingMultiplication(),
    'loglad' : LogRegression(LADRegression(fit_intercept=True)),
    'logols' : LogRegression(sklearn.linear_model.LinearRegression(fit_intercept=True)),
}

METHOD_TYPE_LIST = list(MODEL_TYPES.keys())


def calibrate(values, targets, method='sum', return_coefs=False):
    calibrator = MODEL_TYPES[method]
    feats = values.reshape(-1, 1)
    calibrator.fit(feats, targets)
    fitted = calibrator.predict(feats)
    fitted = np.where(fitted < 0, 0, fitted)
    if return_coefs:
        return fitted, (calibrator.coef_[0], calibrator.intercept_)
    else:
        return fitted


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
parser.add_argument('source_dir',
    help='path to a directory with CSV files with data to be calibrated'
)
parser.add_argument('value_field', nargs='+',
    help='fields in the tested files to be calibrated'
)
parser.add_argument('-i', '--id-fields', nargs='+', default=['id'],
    help='ID column(s) to join the files to the calibration file'
)
parser.add_argument('-t', '--type', nargs='+', default=METHOD_TYPE_LIST,
    help='calibration model type (' + ', '.join(METHOD_TYPE_LIST) + ')'
)
parser.add_argument('-r', '--report-table',
    help='path to output table with calibration coefficients'
)
parser.add_argument('-d', '--target-dir', default=None,
    help='path to a directory to save CSV files with calibrated data (defaults to the source directory)'
)


if __name__ == '__main__':
    args = parser.parse_args()
    calib_df = pd.read_csv(args.calib_file, sep=';')[
        args.id_fields + [args.calib_field]
    ]
    target_dir = args.target_dir if args.target_dir is not None else args.source_dir
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)
    coef_rec = []
    for fname in os.listdir(args.source_dir):
        fname_root, fname_ext = os.path.splitext(fname)
        if fname_ext != '.csv':
            continue
        value_df = pd.read_csv(
            os.path.join(args.source_dir, fname), sep=';'
        ).merge(
            calib_df, on=args.id_fields, how='outer', suffixes=(None, None)
        )
        for model_type in args.type:
            if model_type not in MODEL_TYPES:
                raise ValueError(f'invalid method {model_type}')
            calib_vals = value_df[args.calib_field].fillna(0).values
            for col in args.value_field:
                calibrated, coefs = calibrate(
                    value_df[col].fillna(0).values,
                    calib_vals,
                    method=model_type,
                    return_coefs=True,
                )
                slope, intercept = coefs
                field_name = col + '_cal_' + model_type
                coef_rec.append((fname_root, col, model_type, field_name, slope, intercept))
                print(f'{fname_root} {col} using {model_type}: slope {slope}, intercept {intercept}, field {field_name}')
            value_df[field_name] = calibrated
        value_df[
            value_df[args.id_fields[0]].notna()
        ].drop(
            args.calib_field, axis=1
        ).to_csv(
            os.path.join(args.target_dir, fname_root + '_calib.csv'),
            index=False, sep=';'
        )
    if args.report_table:
        pd.DataFrame.from_records(coef_rec, columns=[
            'file',
            'source_field',
            'model_type',
            'target_field',
            'slope',
            'intercept',
        ]).to_csv(args.report_table, sep=';', index=False)