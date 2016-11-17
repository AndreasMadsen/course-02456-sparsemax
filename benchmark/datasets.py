
from sklearn import datasets
from sklearn.preprocessing import LabelBinarizer


class _AbstractDataset:
    def __init__(self, inputs, targets,
                 regualizer=1e-1, learning_rate=1e-2, name=None):
        self.observations = inputs.shape[0]

        self.input_size = inputs.shape[1]
        self.inputs = inputs

        self.output_size = targets.shape[1]
        self.targets = targets

        self.regualizer = regualizer
        self.learning_rate = learning_rate

        self.name = type(self).__name__.lower() if name is None else name


class Digits(_AbstractDataset):
    def __init__(self):
        digits = datasets.load_digits()

        super().__init__(
            digits.data,
            LabelBinarizer().fit_transform(digits.target)
        )


class Iris(_AbstractDataset):
    def __init__(self):
        iris = datasets.load_iris()

        super().__init__(
            iris.data,
            LabelBinarizer().fit_transform(iris.target)
        )

all_datasets = [Digits, Iris]
