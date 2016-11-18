import _benchmark

import io
import sys
import os.path as path

import subprocess
import numpy as np
import pandas as pd
import scipy.stats

from table import Table

thisdir = path.dirname(path.realpath(__file__))
tabledir = path.join(thisdir, '..', 'latex', 'report', 'tables')
figuredir = path.join(thisdir, '..', 'latex', 'report', 'figures')


def format_best(data, regualizer_values):
    best = regualizer_values[np.argmin(np.mean(data, axis=3), axis=2)]

    exponent = np.floor(np.log10(best))
    mantissa = best / 10**exponent

    return [
        [
            "$%.2f \\cdot 10^{%d}$" % (mantissa_value, exponent_value)
            for mantissa_value, exponent_value in zip(*row)
        ] for row in zip(mantissa, exponent)
    ]


def to_dataframe(data, col_names, row_names, regualizer_values):
    mean = np.mean(data, axis=3)
    sem = scipy.stats.sem(data, axis=3)
    lower, upper = scipy.stats.t.interval(
        0.95, data.shape[3] - 1, loc=mean, scale=sem
    )

    iterables = [col_names, row_names, regualizer_values]
    index = pd.MultiIndex.from_product(
        iterables,
        names=['regressors', 'datasets', 'regualizer']
    )

    # http://stackoverflow.com/questions/36853594
    dataframe = pd.concat([
        pd.DataFrame(mean.ravel(), index=index, columns=['mean']),
        pd.DataFrame(lower.ravel(), index=index, columns=['lower']),
        pd.DataFrame(upper.ravel(), index=index, columns=['upper'])
    ], axis=1)

    dataframe.reset_index(inplace=True)
    return dataframe


def run_plot_script(dataframe, output_file):
    r_script_path = path.join(thisdir, 'plot_hyperparameter.R')

    csv_file = io.StringIO()
    dataframe.to_csv(csv_file)

    p = subprocess.run(
        ['Rscript', r_script_path, output_file],
        input=bytes(csv_file.getvalue(), 'utf8'),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        check=True
    )


def main():
    results = np.load(path.join(tabledir, 'hyperparameter.npz'))

    data = results['data']
    col_names = results['col_names']
    row_names = results['row_names']
    regualizer_values = results['regualizer_values']

    # create table
    content = format_best(data, regualizer_values)
    table = Table(content, col_names, row_names)
    table.save(path.join(tabledir, 'hyperparameter.tex'))

    # create figure
    dataframe = to_dataframe(data, col_names, row_names, regualizer_values)
    run_plot_script(dataframe, path.join(figuredir, 'hyperparameter.pdf'))


if __name__ == "__main__":
    main()
