import _benchmark

import time
import os.path as path

import numpy as np

import datasets
import regressors
from table import SummaryTable
from model_evaluator import ModelEvaluator

thisdir = path.dirname(path.realpath(__file__))
tabledir = path.join(thisdir, '..', 'latex', 'report', 'tables')


def results(regressors, datasets, epochs=1000, n_splits=5, verbose=False):
    '''Saves timings for regressors to filename.txt'''

    col_names = [''] * len(regressors)
    row_names = [''] * len(datasets)
    results = np.zeros((len(datasets), len(regressors), n_splits))

    for dataset_i, DatasetInitializer in enumerate(datasets):
        # intialize dataset
        dataset = DatasetInitializer()
        row_names[dataset_i] = dataset.name
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
            col_names[regressor_i] = regression.name
            if verbose:
                print('  ' + regression.name)

            with regression as model:
                evaluator = ModelEvaluator(
                    model, dataset,
                    epochs=min(epochs, dataset.epochs),
                    random_state=42,
                    verbose=verbose
                )
                missrates = evaluator.all_folds(
                    n_splits=n_splits,
                    stratified=dataset.stratified
                )
                results[dataset_i, regressor_i, :] = missrates

    return (results, col_names, row_names)


def main():
    data, col_names, row_names = results(
        regressors.data_regressors, datasets.all_datasets, verbose=True
    )
    np.savez(
        path.join(tabledir, 'results.npz'),
        data=data, col_names=col_names, row_names=row_names
    )

    table = SummaryTable(data, col_names, row_names)
    table.save(path.join(tabledir, 'results.tex'))

if __name__ == "__main__":
    main()
