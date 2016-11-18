
import os

import tensorflow as tf

import tensorflow_python
import tensorflow_sparsemax
import python_reference
import tensorflow_softmax


class SparsemaxRegressionNativeCPU(tensorflow_sparsemax.SparsemaxRegression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, cpu_only=True, **kwargs)
        self.name = 'TF CPU'


class SparsemaxRegressionNativeGPU(tensorflow_sparsemax.SparsemaxRegression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'TF GPU'


class SparsemaxRegression(tensorflow_sparsemax.SparsemaxRegression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'Sparsemax'

all_regressors = [
    tensorflow_softmax.SoftmaxRegression,
    python_reference.SparsemaxRegression,
    tensorflow_python.SparsemaxRegression,
    SparsemaxRegressionNativeCPU,
    SparsemaxRegressionNativeGPU
]

if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    all_regressors.remove(SparsemaxRegressionNativeGPU)

data_regressors = [
    tensorflow_softmax.SoftmaxRegression,
    SparsemaxRegression
]
