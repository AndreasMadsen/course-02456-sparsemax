
import collections
import os
import os.path as path
import sys
import io
import shutil
import gzip

import numpy as np
import scipy.io.arff
import rarfile
import idx2numpy
import urllib.request
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

thisdir = path.dirname(path.realpath(__file__))
data_home = path.join(thisdir, '..', 'data')

if shutil.which("unrar") is None:
    raise EnvironmentError("unrar is not installed")

Regualizer = collections.namedtuple('Regualizer', ['softmax', 'sparsemax'])
DataPair = collections.namedtuple('DataPair', ['inputs', 'targets'])


class _AbstractDataset:
    def __init__(self, full, train, test,
                 stratified=True,
                 regualizer=Regualizer(softmax=1e-1, sparsemax=1e-1),
                 learning_rate=1e-2, epochs=1000,
                 name=None):
        self.input_size = full.inputs.shape[1]
        self.output_size = full.targets.shape[1]

        self.full = full
        self.test = test
        self.train = train

        self.stratified = stratified
        self.regualizer = regualizer
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.name = type(self).__name__ if name is None else name


def _download_mnist_file(name):
    data_root = 'http://yann.lecun.com/exdb/mnist/'

    href = data_root + name + '.gz'
    data_dir = path.join(data_home, 'mnist')
    file_gz = path.join(data_dir, name + '.gz')

    # data have not been feched
    if not path.exists(file_gz):
        os.makedirs(data_dir, exist_ok=True)

        # download file
        urllib.request.urlretrieve(href, file_gz)

    with gzip.open(file_gz, 'rb') as f:
        ndarray = idx2numpy.convert_from_file(f)

    return ndarray


class MNIST(_AbstractDataset):
    def __init__(self):
        file_npz = path.join(data_home, 'mnist', 'mnist.npz')

        # data have not been feched and processed
        if not path.exists(file_npz):
            train_inputs = _download_mnist_file('train-images-idx3-ubyte')
            train_targets = _download_mnist_file('train-labels-idx1-ubyte')
            test_inputs = _download_mnist_file('t10k-images-idx3-ubyte')
            test_targets = _download_mnist_file('t10k-labels-idx1-ubyte')

            # convert inputs
            train_inputs = train_inputs.reshape(train_inputs.shape[0], 784)
            test_inputs = test_inputs.reshape(test_inputs.shape[0], 784)

            # convert targets
            train_targets = LabelBinarizer().fit_transform(train_targets)
            test_targets = LabelBinarizer().fit_transform(test_targets)

            # save inputs and targets numpy arrays
            np.savez(file_npz,
                     train_inputs=train_inputs,
                     train_targets=train_targets,
                     test_inputs=test_inputs, test_targets=test_targets)

        # data have all ready been processed, just load it
        data = np.load(file_npz)

        super().__init__(
            full=DataPair(data['train_inputs'], data['train_targets']),
            train=DataPair(data['train_inputs'], data['train_targets']),
            test=DataPair(data['test_inputs'], data['test_targets']),
            regualizer=Regualizer(softmax=1e-7, sparsemax=1e-6),
            epochs=100,
            stratified=True
        )


class Iris(_AbstractDataset):
    def __init__(self):
        iris = datasets.load_iris()

        targets = LabelBinarizer().fit_transform(iris.target)

        train_inputs, test_inputs, train_targets, test_targets = \
            train_test_split(
                iris.data, targets,
                test_size=0.1, random_state=42, stratify=iris.target
            )

        super().__init__(
            full=DataPair(iris.data, targets),
            train=DataPair(train_inputs, train_targets),
            test=DataPair(test_inputs, test_targets),
            regualizer=Regualizer(softmax=1e-8, sparsemax=1e-8),
            stratified=True
        )


def _arff_to_data_pair(file_arff):
    # load arff file
    with open(file_arff, 'r') as f:
        data, meta = scipy.io.arff.loadarff(f)

    # get inputs names
    target_names = [
        name
        for name, type in zip(meta.names(), meta.types())
        if (type is 'nominal')
    ]

    input_names = [
        name for name in meta.names() if name not in target_names
    ]

    # stack collums in a 2d-numpy array, for both inputs and targets
    inputs = np.vstack(
        [data[name] for name in input_names]
    ).astype(np.float64).T

    targets = np.vstack(
        [data[name] for name in target_names]
    ).astype(np.float64).T

    # convert targets to a distribution
    targets /= targets.sum(axis=1)[:, np.newaxis]

    return (inputs, targets)


def _mulan(name, pre_split=False):
    data_root = 'http://sourceforge.net/projects/mulan/files/datasets/'

    # prepear
    href = data_root + name + '.rar'
    data_dir = path.join(data_home, name)
    file_rar = path.join(data_dir, name + '.rar')
    file_npz = path.join(data_dir, name + '.npz')

    file_arff = path.join(data_dir, name + '.arff')
    file_train_arff = path.join(data_dir, name + '-train.arff')
    file_test_arff = path.join(data_dir, name + '-test.arff')

    # data have not been feched and processed
    if not path.exists(file_npz):
        os.makedirs(data_dir, exist_ok=True)

        # download file
        urllib.request.urlretrieve(href, file_rar)

        # extract arff file from rar archive
        with rarfile.RarFile(file_rar) as rf:
            rf.extract(name + '.arff', path=data_dir)
            if pre_split:
                rf.extract(name + '-train.arff', path=data_dir)
                rf.extract(name + '-test.arff', path=data_dir)

        full_inputs, full_targets = _arff_to_data_pair(file_arff)
        if pre_split:
            train_inputs, train_targets = _arff_to_data_pair(file_train_arff)
            test_inputs, test_targets = _arff_to_data_pair(file_test_arff)

        # save inputs and targets numpy arrays
        if pre_split:
            np.savez(file_npz,
                     full_inputs=full_inputs, full_targets=full_targets,
                     train_inputs=train_inputs, train_targets=train_targets,
                     test_inputs=test_inputs, test_targets=test_targets)
        else:
            np.savez(file_npz,
                     full_inputs=full_inputs, full_targets=full_targets)

    # data have all ready been processed, just load it
    data = np.load(file_npz)

    # create data pair
    full = DataPair(data['full_inputs'], data['full_targets'])
    if pre_split:
        train = DataPair(data['train_inputs'], data['train_targets'])
        test = DataPair(data['test_inputs'], data['test_targets'])
    else:
        train_inputs, test_inputs, train_targets, test_targets = \
            train_test_split(
                data['full_inputs'], data['full_targets'],
                test_size=0.1, random_state=42
            )
        train = DataPair(train_inputs, train_targets)
        test = DataPair(test_inputs, test_targets)

    return (full, train, test)


class Scene(_AbstractDataset):
    def __init__(self):
        full, train, test = _mulan('scene', pre_split=True)
        super().__init__(
            full, train, test,
            regualizer=Regualizer(softmax=1e-8, sparsemax=1e-4),
            stratified=False
        )


class Emotions(_AbstractDataset):
    def __init__(self):
        full, train, test = _mulan('emotions', pre_split=True)
        super().__init__(
            full, train, test,
            regualizer=Regualizer(softmax=1e-2, sparsemax=1e-2),
            stratified=False
        )


class CAL500(_AbstractDataset):
    def __init__(self):
        full, train, test = _mulan('CAL500', pre_split=False)
        super().__init__(
            full, train, test,
            regualizer=Regualizer(softmax=1e-8, sparsemax=1e-1),
            stratified=False
        )

all_datasets = [MNIST, Iris, Scene, Emotions, CAL500]
