import _benchmark

import time
import os.path as path

import numpy as np

import datasets
import regressors
from table import Table
from model_evaluator import ModelEvaluator

thisdir = path.dirname(path.realpath(__file__))


def results(regressors, datasets, epochs=1000, n_splits=5, verbose=False):
    '''Saves timings for regressors to filename.txt'''

    col_names = [''] * len(datasets)
    row_names = [''] * len(regressors)
    results = np.zeros((len(regressors), len(datasets), n_splits))

    for dataset_i, DatasetInitializer in enumerate(datasets):
        # intialize dataset
        dataset = DatasetInitializer()
        col_names[dataset_i] = dataset.name
        if verbose:
            print(dataset.name)

        for regressor_i, regressor in enumerate(regressors):
            # intialize model
            regression = regressor(
                input_size=dataset.input_size,
                output_size=dataset.output_size,
                random_state=42,
                regualizer=dataset.regualizer,
                learning_rate=dataset.learning_rate
            )
            row_names[regressor_i] = regression.name
            if verbose:
                print('  ' + regression.name)

            with regression as model:
                evaluator = ModelEvaluator(
                    model, dataset,
                    random_state=42, epochs=epochs,
                    verbose=verbose
                )
                missrates = evaluator.all_folds(n_splits=n_splits)
                results[regressor_i, dataset_i, :] = missrates

    return (results, col_names, row_names)


def main():
    data, col_names, row_names = results(
        regressors.data_regressors, datasets.all_datasets, epochs=1000, verbose=True
    )
    table = Table(data, col_names, row_names)
    table.save(
        path.join(thisdir, '..', 'latex', 'report', 'tables', 'results.tex')
    )

if __name__ == "__main__":
    main()
