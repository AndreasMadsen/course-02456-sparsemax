import _benchmark

import time
import os.path as path

import numpy as np

import datasets
import regressors
from table import PairTable
from model_evaluator import ModelEvaluator

thisdir = path.dirname(path.realpath(__file__))
tabledir = path.join(thisdir, '..', 'latex', 'report', 'tables')


def results(regressors, datasets, epochs=1000, verbose=False):
    '''Saves timings for regressors to filename.txt'''

    col_names = [''] * len(regressors)
    row_names = [''] * len(datasets)
    results = np.zeros((len(datasets), len(regressors), 2))

    for dataset_i, DatasetInitializer in enumerate(datasets):
        # intialize dataset
        dataset = DatasetInitializer()
        row_names[dataset_i] = dataset.name
        if verbose:
            print(dataset.name)

        for regressor_i, Regressor in enumerate(regressors):
            # intialize model
            regualizer = getattr(dataset.regualizer, Regressor.transform_type)
            regression = Regressor(
                input_size=dataset.input_size,
                output_size=dataset.output_size,
                random_state=42,
                regualizer=regualizer,
                learning_rate=dataset.learning_rate
            )
            col_names[regressor_i] = regression.name
            if verbose:
                print('  ' + regression.name)

            with regression as model:
                model.reset()
                model.update(dataset.train.inputs, dataset.train.targets,
                             epochs=min(epochs, dataset.epochs))

                divergence = ModelEvaluator.evaluate(model,
                                                     dataset.test.inputs,
                                                     dataset.test.targets)
                results[dataset_i, regressor_i, 0] = divergence

                if dataset.multi_class:
                    missrate = np.nan
                else:
                    missrate = model.error(
                        dataset.test.inputs, dataset.test.targets
                    )
                results[dataset_i, regressor_i, 1] = missrate

                if verbose:
                    print('    %f / %f' % (divergence, missrate))

    return (results, col_names, row_names)


def format_table(fn, data):
    return [
        [
            fn(data_value) for data_value in data_row
        ] for data_row in data
    ]


def main():
    """data, col_names, row_names = results(
        regressors.data_regressors, datasets.all_datasets, verbose=True
    )
    np.savez(
        path.join(tabledir, 'results.npz'),
        data=data, col_names=col_names, row_names=row_names
    )"""
    results = np.load(path.join(tabledir, 'results.npz'))
    data = results['data']
    col_names = results['col_names']
    row_names = results['row_names']

    divergence = format_table(
        lambda val: "%.3f" % val,
        data[:, :, 0]
    )
    missrate = format_table(
        lambda val: '--' if np.isnan(val) else '$%.1f \\%%$' % (val * 100),
        data[:, :, 1]
    )

    table = PairTable(
        divergence, missrate,
        col_names, ['$\mathbf{JS}$', 'error rate'], row_names
    )
    table.save(path.join(tabledir, 'results.tex'))

if __name__ == "__main__":
    main()
