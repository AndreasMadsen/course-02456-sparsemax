import IPython
import _benchmark
import sys
import time
import numpy as np

from sklearn import datasets

from model_test import ModelTest

from tensorflow_python import SparsemaxRegression as SparseMaxTFPython
from tensorflow_sparsemax import SparsemaxRegression as SparseMaxTF
from python_reference import SparsemaxRegression as SparseMaxPython
from tensorflow_softmax import SoftmaxRegression as SoftMaxTF

from sklearn.preprocessing import LabelBinarizer

from tabulate import tabulate

def computeStats(results):
    n = len(results)
    mean = round(np.mean(results),2)
    se = np.std(results)/np.sqrt(n)
    ci = round(1.96*se,2)
    return str(mean) + "+-" + str(ci)


def computeResults(regressors, regressors_tex, datasets,
                   datasets_tex, filename):
    ''' Saves results for regressors to filename.txt'''
    # Initialize empty list and dict for storing results
    results = []

    # Loop through all regressors and datasets and compute the avg_missrate
    # Save as .txt file with a .tex formated table by use of tabulate
    for regressor, regressor_name in zip(regressors, regressors_tex):
        # Initialize empty list for storing model specific results
        results_data = []
        for dataset, dataset_name in zip(datasets, datasets_tex):
            # Get sizes
            input_size = dataset.data.shape[1]
            output_size = max(dataset.target) + 1

            # intialize model
            regression = regressor(
                input_size=input_size, output_size=output_size,
                random_state=42, regualizer=1e-1, learning_rate=1e-2
            )

            with regression as model:
                # setup model tester
                tester = ModelTest(model, dataset, verbose=False, random_state=42)
                result = tester.test(0.2, 0.1)
                results_data.append(result)
        tmp = [regressor_name]
        tmp.extend(results_data)
        results.append(tmp)

    # Save results
    headers = ["Model"]
    headers.extend(datasets_tex)
    with open(filename, "w") as text_file:
        text_file.write(tabulate(results, tablefmt="latex",
                                          floatfmt = ".3f",
                                          headers=headers))

def computeTimings(regressors, regressors_tex, datasets,
                   datasets_tex, filename, no_calls = 50):
    '''Saves timings for regressors to filename.txt'''

    elapsed_times = []
    results_times = []
    for dataset, dataset_name in zip(datasets, datasets_tex):
        stats = []
        targets = LabelBinarizer().fit_transform(dataset.target)
        for regressor, regressor_name in zip(regressors, regressors_tex):
            # Get sizes
            input_size = dataset.data.shape[1]
            output_size = max(dataset.target) + 1

            # intialize model
            regression = regressor(
                input_size=input_size, output_size=output_size,
                random_state=42, regualizer=1e-1, learning_rate=1e-2
            )

            with regression as model:
                for _ in range(no_calls):
                    model.reset()
                    t = time.perf_counter()
                    for epoch in range(1000):
                        model.update(dataset.data, targets)
                    elapsed_times.append(time.perf_counter() - t)
                stats.append(computeStats(elapsed_times))
        tmp = [dataset_name]
        tmp.extend(stats)
        results_times.append(tmp)

    headers = ["Dataset"]
    headers.extend(regressors_tex)
    with open(filename, 'w') as text_file:
         text_file.write(tabulate(
            results_times,
            tablefmt="latex",
            headers=headers))



def main():
    # Define list of regressors
    #regressors = [SparseMaxTFPython, SparseMaxTF, SparseMaxPython, SoftMaxTF]
    regressors = [SparseMaxTF, SoftMaxTF]
    # Define names to be diplayed in .tex table
    #regressors_tex = ["TensorFlow Python", "TensorFlow", "Numpy", "SoftMax"]
    regressors_tex = ["SparseMax","SoftMax"]
    # Get data
    digits = datasets.load_digits()
    iris = datasets.load_iris()

    # Define list of datasets
    lDataset = [digits, iris]

    # Define names of datasets to be displayed in .tex table
    dataset_tex = ["MNIST", "Iris"]

    computeResults(regressors, regressors_tex, lDataset, dataset_tex,
                   "results.txt")

    computeTimings(regressors, regressors_tex, lDataset, dataset_tex,
                   "timings.txt", no_calls=2)

if __name__ == "__main__":
    main()
