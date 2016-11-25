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


def hyperparameter(regressors, datasets, regualizer_values,
                   epochs=1000, n_splits=5, verbose=False):

    col_names = [''] * len(regressors)
    row_names = [''] * len(datasets)
    results = np.zeros(
        (len(datasets), len(regressors), len(regualizer_values), n_splits)
    )

    for dataset_i, DatasetInitializer in enumerate(datasets):
        # intialize dataset
        dataset = DatasetInitializer()
        row_names[dataset_i] = dataset.name
        if verbose:
            print(dataset.name)

        for regressor_i, Regressor in enumerate(regressors):
            for regualizer_i, regualizer in enumerate(regualizer_values):
                # intialize model
                regression = Regressor(
                    input_size=dataset.input_size,
                    output_size=dataset.output_size,
                    random_state=42,
                    regualizer=regualizer,
                    learning_rate=dataset.learning_rate
                )
                col_names[regressor_i] = regression.name
                if regualizer_i == 0 and verbose:
                    print('  ' + regression.name)

                with regression as model:
                    evaluator = ModelEvaluator(
                        model, dataset.train,
                        epochs=min(epochs, dataset.epochs),
                        random_state=42
                    )
                    divergence = evaluator.all_folds(
                        n_splits=n_splits,
                        stratified=dataset.stratified
                    )
                    if verbose:
                        print('    %e: %f' % (regualizer, np.mean(divergence)))

                    results[dataset_i, regressor_i, regualizer_i, :] = \
                        divergence

    return (results, col_names, row_names)


def main():
    regualizer_values = 10.0**np.arange(-8, 3)

    data, col_names, row_names = hyperparameter(
        regressors.data_regressors, datasets.all_datasets, regualizer_values,
        verbose=True
    )

    np.savez(
        path.join(tabledir, 'hyperparameter.npz'),
        data=data, col_names=col_names, row_names=row_names,
        regualizer_values=regualizer_values
    )

if __name__ == "__main__":
    main()
