import _benchmark

import time
import os.path as path

import numpy as np

import datasets
import regressors
from table import Table

thisdir = path.dirname(path.realpath(__file__))


def timings(regressors, datasets, epochs=1000, iterations=50, verbose=False):
    '''Saves timings for regressors to filename.txt'''

    col_names = [''] * len(datasets)
    row_names = [''] * len(regressors)
    results = np.zeros((len(regressors), len(datasets), iterations))

    for dataset_i, DatasetInitializer in enumerate(datasets):
        # intialize dataset
        dataset = DatasetInitializer()
        col_names[dataset_i] = dataset.name
        if verbose:
            print(dataset.name)

        for regressor_i, regressor in enumerate(regressors):
            # intialize model
            regression = regressor(
                observations=dataset.observations,
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
                for iteration_i in range(iterations):
                    # reset model and train model
                    model.reset()

                    tick = time.perf_counter()
                    model.update(
                        dataset.inputs, dataset.targets, epochs=epochs
                    )
                    tock = time.perf_counter() - tick
                    results[regressor_i, dataset_i, iteration_i] = tock
                    if verbose:
                        print('      %d: %f' % (iteration_i, tock))

    return (results, col_names, row_names)


def main():
    data, col_names, row_names = timings(
        regressors.all_regressors, datasets.all_datasets
    )
    table = Table(data, col_names, row_names)
    table.save(
        path.join(thisdir, '..', 'latex', 'report', 'tables', 'timings.tex')
    )

if __name__ == "__main__":
    main()
