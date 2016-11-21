import _benchmark

import time
import os.path as path

import numpy as np

import datasets
import regressors
from table import SummaryTable

thisdir = path.dirname(path.realpath(__file__))
tabledir = path.join(thisdir, '..', 'latex', 'report', 'tables')


def timings(regressors, datasets, epochs=100, iterations=10, verbose=False):
    '''Saves timings for regressors to filename.txt'''

    col_names = [''] * len(regressors)
    row_names = [''] * len(datasets)
    results = np.zeros((len(datasets), len(regressors), iterations))

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
                observations=dataset.observations,
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
                for iteration_i in range(iterations):
                    # reset model and train model
                    model.reset()

                    tick = time.perf_counter()
                    model.update(
                        dataset.inputs, dataset.targets, epochs=epochs
                    )
                    tock = time.perf_counter() - tick
                    results[dataset_i, regressor_i, iteration_i] = tock
                    if verbose:
                        print('      %d: %f' % (iteration_i, tock))

    return (results, col_names, row_names)


def main():
    data, col_names, row_names = timings(
        regressors.all_regressors, datasets.all_datasets, verbose=True
    )
    np.savez(
        path.join(tabledir, 'timings.npz'),
        data=data, col_names=col_names, row_names=row_names
    )

    table = SummaryTable(data, col_names, row_names, format="$%.2f \\pm %.3f$")
    table.save(path.join(tabledir, 'timings.tex'))

if __name__ == "__main__":
    main()
