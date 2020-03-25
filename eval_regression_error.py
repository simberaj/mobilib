import argparse
import io
import os
import math
import base64

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPORT_TEMPLATE = '''<!DOCTYPE html>
<html>
<head>
  <meta charset="utf8">
  <title>Validation result</title>
  <style type="text/css">
    body {font-family: "Lucida Sans Unicode", "Lucida Grande", sans-serif; font-size: 11px;}
    h1 {font-size: 150%; text-align: center;}
    h2 {font-size: 125%; text-align: center;}
    table {margin-left:auto; margin-right:auto;}
    img {display: block; margin-left:auto; margin-right:auto;}
    tr:nth-child(even) {background-color: #f2f2f2}
    th, td {padding: 5px;}
    th {text-align: left;}
    td {text-align: right; padding: 5px;}
  </style>
</head>
<body>
  <h1>Validation result</h1>
    <h2>Dataset characteristics</h2>
      {{setdescstat}}
    <h2>Error measures</h2>
      {{erragg}}
    <h2>Correlation plot</h2>
    <img src="data:image/png;base64,{{imgdump}}" alt="Matpllotlib unavailable or image lost"/>
</body>
</html>
'''


def evaluate(preds, trues, report_path=None, round=None):
    validator = Validator(preds, trues, round=round)
    validator.validate()
    # for ind, val in validator.results.items():
        # print(ind.ljust(7), val)
    if report_path:
        validator.output(report_path)
    return validator.results


def dict_to_table(data, heads, rowHead=False):
    if isinstance(data, list):
        data = {key : [str(item[key]) for item in data] for key in data[0]}
    else:
        data = {key : [str(value)] for key, value in data.items()}
    rows = []
    for key, name in heads:
        rows.append([name] + data[key])
    return list_to_table(rows, colHead=True, rowHead=rowHead)

def list_to_table(rows, colHead=False, rowHead=False):
    htmlRows = []
    htmlRows = [row_to_html(rows[0], 'th')] if rowHead else []
    for row in rows[(1 if rowHead else 0):]:
        htmlRows.append(row_to_html(row, 'td', 'th' if colHead else None))
    return '<table><tbody><tr>' + '</tr><tr>'.join(htmlRows) + '</tr></tbody></table>'

def row_to_html(row, tag, firstTag=None):
    if firstTag:
        return start_tag(firstTag) + row[0] + end_tag(firstTag) + row_to_html(row[1:], tag)
    else:
        start = start_tag(tag)
        end = end_tag(tag)
        return start + (end + start).join(row) + end

def start_tag(tag):
    return '<' + tag + '>'

def end_tag(tag):
    return '</' + tag + '>'


class Validator:
    ERRAGG_NAMES = [
        ('mismatch', 'Absolute value sum difference'),
        ('rmse', 'Root mean square error (RMSE)'),
        ('tae', 'Total absolute error (TAE)'),
        ('rtae', 'Relative total absolute error (RTAE)'),
        ('r2', 'Coefficient of determination (R<sup>2</sup>)'),
        ('cp', 'Common part'),
        ('log_r2', 'Coefficient of determination (R<sup>2</sup>) on log10-transformed values'),
    ]
    DESC_NAMES = [
        ('set', 'Dataset'),
        ('sum', 'Sum'),
        ('min', 'Minimum'),
        ('q2l', 'Q2,5'),
        ('q10', 'Q10 - 1st Decile'),
        ('q25', 'Q25 - 1st Quartile'),
        ('median', 'Q50 - Median'),
        ('q75', 'Q75 - 3rd Quartile'),
        ('q90', 'Q90 - 9th Decile'),
        ('q2h', 'Q97,5'),
        ('max', 'Maximum'),
        ('mean', 'Mean'),
    ]
    FORMATS = {
        'mismatch' : '{:g}',
        'tae' : '{:.0f}',
        'rtae' : '{:.2%}',
        'r2' : '{:.3%}',
        'rmse' : '{:.2f}',
        'cp' : '{:.3%}',
        'log_r2' : '{:.3%}',
    }
    FORMATS.update({item[0] : '{:g}' for item in DESC_NAMES})

    def __init__(self, models, reals, round=None):
        self.models = models
        self.reals = reals
        self.round = round

    def validate(self):
        self.realSum = self.reals.sum()
        self.realMean = self.realSum / len(self.reals)
        self.modelSum = self.models.sum()
        self.mismatch = self.modelSum - self.realSum
        self.resid = self.models - self.reals
        self.absResid = abs(self.resid)
        self.sqResidSum = (self.absResid ** 2).sum()
        self.tae = self.absResid.sum()
        self.rtae = self.tae / self.realSum
        self.r2 = 1 - self.sqResidSum / ((self.reals - self.realMean) ** 2).sum()
        self.rmse = math.sqrt(self.sqResidSum / len(self.reals))
        self.cp = 2 * np.fmin(self.models, self.reals).sum() / (self.realSum + self.modelSum)
        log_models = np.log10(self.models + 1)
        log_reals = np.log10(self.reals + 1)
        self.logr2 = 1 - (abs(log_models - log_reals) ** 2).sum() / ((log_reals - log_reals.mean()) ** 2).sum()

    @property
    def results(self):
        return {
            'DIFF' : self.mismatch,
            'TAE' : self.tae,
            'RTAE' : self.rtae,
            'R2' : self.r2,
            'RMSE' : self.rmse,
            'CP': self.cp,
            'LOGR2': self.logr2,
        }

    def describe(self, data):
        return {
            'min' : data.min(),
            'max' : data.max(),
            'sum' : data.sum(),
            'mean' : data.mean(),
            'median' : np.median(data),
            'q25' : np.percentile(data, 25),
            'q75' : np.percentile(data, 75),
            'q10' : np.percentile(data, 10),
            'q90' : np.percentile(data, 90),
            'q2l' : np.percentile(data, 2.5),
            'q2h' : np.percentile(data, 97.5)
        }

    def descriptions(self):
        descs = []
        for name, data in (
                ('Modeled', self.models),
                ('Real', self.reals),
                ('Residuals', self.resid),
                ('Abs(Residuals)', self.absResid)
            ):
            desc = self.format(self.describe(data))
            desc['set'] = name
            descs.append(desc)
        return descs

    def globals(self):
        return dict(
            mismatch=self.mismatch,
            tae=self.tae,
            rtae=self.rtae,
            r2=self.r2,
            rmse=self.rmse,
            cp=self.cp,
            log_r2=self.logr2,
        )

    def format(self, fdict):
        return {
            key : self.FORMATS[key].format(float(value)).replace('.', ',')
            for key, value in fdict.items()
        }

    def output(self, fileName):
        template = REPORT_TEMPLATE.replace('{', '[') \
                                  .replace('}', ']') \
                                  .replace('[[', '{') \
                                  .replace(']]', '}')
        fileNameBase = os.path.splitext(fileName)[0]
        text = template.format(
            erragg=dict_to_table(
                self.format(self.globals()),
                self.ERRAGG_NAMES
            ),
            setdescstat=dict_to_table(
                self.descriptions(),
                self.DESC_NAMES,
                rowHead=True
            ),
            imgdump=base64.b64encode(self.image()).decode('ascii'),
        )
        with open(fileName, 'w') as outfile:
            outfile.write(text.replace('[', '{').replace(']', '}'))

    def image(self):
        plt.figure()
        plot_reals = (
            self.reals if self.round is None
            else np.around(self.reals, self.round)
        )
        plot_models = (
            self.models if self.round is None
            else np.around(self.models, self.round)
        )
        plt.loglog(plot_reals, plot_models, 'b.')
        ax = plt.gca()
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]
        ax.plot(lims, lims, 'r-', alpha=0.25, zorder=0)
        ax.set_aspect('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        plt.xlabel('Real values')
        plt.ylabel('Modeled values')
        dump = io.BytesIO()
        plt.savefig(dump, bbox_inches='tight')
        plt.close()
        return dump.getvalue()


parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('files_dir',
    help='path to a directory with CSV files with data to be tested'
)
parser.add_argument('test_file',
    help='path to CSV with testing (ground truth) data'
)
parser.add_argument('test_field',
    help='field in the testing data to evaluate against'
)
parser.add_argument('-f', '--value-field', nargs='+',
    help='field(s) in the tested files to be evaluated'
)
parser.add_argument('-F', '--all-other-fields', action='store_true',
    help='evaluate all field(s) in the tested files except ID fields'
)
parser.add_argument('-a', '--agg-across-ids',
    help='aggregate all items for identical ID(s) before validating by this function'
)
parser.add_argument('-i', '--id-fields', nargs='+', default=['id'],
    help='ID column(s) to join the tested files to the testing file'
)
parser.add_argument('-m', '--metric-file',
    help='file to output a CSV with general error metrics'
)
parser.add_argument('-r', '--round', type=int,
    help='round the compared values to this many decimal digits'
)

if __name__ == '__main__':
    args = parser.parse_args()
    test_df = pd.read_csv(args.test_file, sep=';')[
        args.id_fields + [args.test_field]
    ]
    if args.agg_across_ids:
        test_df = test_df.groupby(args.id_fields)[args.test_field].agg(
            args.agg_across_ids
        ).reset_index()
    metric_recs = []
    for fname in os.listdir(args.files_dir):
        fname_root, fname_ext = os.path.splitext(fname)
        if fname_ext != '.csv':
            continue
        value_df = pd.read_csv(os.path.join(args.files_dir, fname), sep=';')
        if args.all_other_fields:
            eval_fields = [fld for fld in value_df if fld not in args.id_fields]
        else:
            eval_fields = args.value_field
        if args.agg_across_ids:
            value_df = value_df.groupby(args.id_fields)[eval_fields].agg(
                args.agg_across_ids
            ).reset_index()
        eval_df = test_df.merge(value_df, on=args.id_fields, how='outer', suffixes=(None, None))
        test_vals = eval_df[args.test_field].fillna(0).to_numpy()
        for value_field in eval_fields:
            model_vals = eval_df[value_field].fillna(0).to_numpy()
            metrics = evaluate(
                model_vals,
                test_vals,
                report_path=os.path.join(
                    args.files_dir, fname_root + f'_{value_field}.html'
                ),
                round=args.round,
            )
            metrics['name'] = fname_root
            metrics['field'] = value_field
            metric_recs.append(metrics)
    metric_df = pd.DataFrame.from_records(metric_recs)
    print(metric_df.head(20))
    if args.metric_file:
        metric_df.to_csv(args.metric_file, sep=';', index=False)
    