
import os

import tensorflow_python
import tensorflow_sparsemax
import python_reference
import tensorflow_softmax

CUDA_VISIBLE_DEVICES = None
if 'CUDA_VISIBLE_DEVICES' in os.environ:
    CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES']


class SparsemaxRegressionNativeCPU(tensorflow_sparsemax.SparsemaxRegression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'TF CPU'

    def __enter__(self):
        if CUDA_VISIBLE_DEVICES is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
        return super().__enter__()

    def __exit__(self, *args):
        if CUDA_VISIBLE_DEVICES is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
        return super().__exit__(*args)


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

if CUDA_VISIBLE_DEVICES is None:
    all_regressors.remove(SparsemaxRegressionNativeGPU)

data_regressors = [
    tensorflow_softmax.SoftmaxRegression,
    SparsemaxRegression
]
