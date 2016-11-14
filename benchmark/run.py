import _benchmark
import sys

from sklearn import datasets

from model_test import ModelTest
# from tensorflow_python import SparsemaxRegression
# from tensorflow_sparsemax import SparsemaxRegression
from python_reference import SparsemaxRegression

# get data
if sys.argv[1] == "iris":
    dataset = datasets.load_iris()
if sys.argv[1] == "mnist":
    dataset = datasets.load_digits()

input_size = dataset.data.shape[1]
output_size = max(dataset.target) + 1

# intialize model
regression = SparsemaxRegression(
    input_size=input_size, output_size=output_size,
    random_state=42, regualizer=1e-1, learning_rate=1e-2
)

with regression as model:
    # setup model tester
    tester = ModelTest(regression, dataset, verbose=True, random_state=42)
    tester.test(loss=0.2, missrate=0.1)
