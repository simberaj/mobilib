"""Show places with gaps in a Zipf-like (power law) distributed data.

These places might be used to determine "natural" bounds to bin the data
or create eligibility criteria.

Creates a Matplotlib figure and shows it.
"""

import os
from typing import Optional, List
from numbers import Number

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model

import mobilib.argparser


def fit_exponent(values: pd.Series,
                 plot: bool = True,
                 ) -> float:
    vals = values.sort_values(ascending=False)
    vals = vals[vals > 1]
    logvals = np.log(vals)
    ords = np.arange(len(vals)) + 1
    logords = np.log(ords)
    logord_preds = logords.reshape(-1, 1)
    mod = sklearn.linear_model.LinearRegression()
    mod.fit(logord_preds, logvals, sample_weight=logvals)
    exponent = -float(mod.coef_[0])
    if plot:
        plt.figure()
        plt.plot(ords, vals, label='True values')
        plt.plot(
            ords, np.exp(mod.predict(logord_preds)) - 1,
            label=f'Predicted values (exponent: {exponent:.4f})'
        )
        plt.xlabel('Value rank (descending)')
        plt.ylabel('Value')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.title(f'{values.name} log-log OLS fit')
        plt.show()
    return exponent


def select_values(values: pd.Series,
                  from_value: Optional[Number] = None,
                  to_value: Optional[Number] = None,
                  by_index: bool = False,
                  ) -> pd.Series:
    if by_index:
        is_selected = np.ones(len(values), dtype=bool)
    else:
        is_selected = pd.Series(True, index=values.index)
    selector = values.index if by_index else values
    if from_value is not None:
        is_selected &= (selector >= from_value)
    if to_value is not None:
        is_selected &= (selector <= to_value)
    return values[is_selected]


def calculate_zipf_gaps(values: pd.Series,
                        exponent: Number = 1,
                        spread: Number = 1,
                        ) -> pd.Series:
    if spread < 1:
        spread = int(round(len(values) * spread))
    sorted_vals = values.sort_values(ascending=False)
    val_arr = sorted_vals.to_numpy()
    val_arr = val_arr[val_arr > 0]
    ranks = np.arange(len(val_arr)-spread) + 1
    first_rank = spread // 2
    last_rank = len(val_arr) - spread + first_rank
    fractions = val_arr[:-spread] / val_arr[spread:]
    measure = fractions ** exponent * ranks / (ranks + spread)
    return pd.Series(
        measure,
        index=val_arr[first_rank:last_rank],
        name=values.name,
    ).rolling(spread, min_periods=1, center=True).mean()


def plot_output(measures: List[pd.Series],
                from_value: Optional[Number] = None,
                to_value: Optional[Number] = None,
                ) -> None:
    plt.figure()
    for measure in measures:
        disp_measure = select_values(measure, from_value, to_value, by_index=True)
        plt.plot(disp_measure.index, disp_measure.to_numpy(), label=measure.name)
    plt.legend()
    plt.show()
    

parser = mobilib.argparser.default(__doc__)
parser.add_argument(
    '-s', '--source', nargs='+', action='append',
    help='path to semicolon-delimited CSV file followed by names of columns to use'
)
parser.add_argument(
    '-F', '--from-value', type=float,
    help='minimum value to consider'
)
parser.add_argument(
    '-T', '--to-value', type=float,
    help='maximum value to consider'
)
parser.add_argument(
    '-f', '--disp-from-value', type=float,
    help='minimum value to display'
)
parser.add_argument(
    '-t', '--disp-to-value', type=float,
    help='maximum value to display'
)
parser.add_argument(
    '-e', '--exponent', type=float,
    help='apriori value for power law distribution exponent (default: determine by log-log OLS fit)'
)
parser.add_argument(
    '-g', '--gap', type=int, default=1,
    help='gap size to evaluate measure'
)
parser.add_argument(
    '-q', '--quiet', action='store_true',
    help='do not show log-log OLS plots for exponent derivation'
)


if __name__ == '__main__':
    args = parser.parse_args()
    gap_measures = []
    for source_def in args.source:
        fpath = source_def[0]
        fname = os.path.splitext(fpath)[0]
        df = pd.read_csv(fpath, sep=';')
        try:
            coef = float(source_def[-1])
        except ValueError:
            coef = 1
        else:
            source_def = source_def[:-1]
        for colname in source_def[1:]:
            comp_values = select_values(df[colname] * coef, args.from_value, args.to_value)
            comp_values.rename(f'{fname}_{colname}', inplace=True)
            # series.append(values)
            comp_exponent = args.exponent
            if not comp_exponent:
                comp_exponent = fit_exponent(comp_values, plot=(not args.quiet))
            # exponents.append(exponent)
            gap_measures.append(
                calculate_zipf_gaps(comp_values, comp_exponent, spread=args.gap)
            )
    plot_output(gap_measures, args.disp_from_value, args.disp_to_value)
