
import tensorflow_python
import tensorflow_sparsemax
import python_reference
import tensorflow_softmax

all_regressors = [
    tensorflow_python.SparsemaxRegression,
    tensorflow_sparsemax.SparsemaxRegression,
    python_reference.SparsemaxRegression,
    tensorflow_softmax.SoftmaxRegression
]

data_regressors = [
    tensorflow_sparsemax.SparsemaxRegression,
    tensorflow_softmax.SoftmaxRegression
]
