
import os
import os.path as path
import sys
import io
import shutil

import numpy as np
import scipy.io.arff
import rarfile
import urllib.request
from sklearn import datasets
from sklearn.preprocessing import LabelBinarizer

thisdir = path.dirname(path.realpath(__file__))
data_home = path.join(thisdir, '..', 'data')

if shutil.which("unrar") is None:
    raise EnvironmentError("unrar is not installed")


class _AbstractDataset:
    def __init__(self, inputs, targets,
                 stratified=True,
                 regualizer=1e-1, learning_rate=1e-2, epochs=1000,
                 name=None):
        self.observations = inputs.shape[0]

        self.input_size = inputs.shape[1]
        self.inputs = inputs

        self.output_size = targets.shape[1]
        self.targets = targets

        self.stratified = stratified
        self.regualizer = regualizer
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.name = type(self).__name__ if name is None else name


class MNIST(_AbstractDataset):
    def __init__(self):
        digits = datasets.fetch_mldata('MNIST original', data_home=data_home)

        super().__init__(
            digits.data,
            LabelBinarizer().fit_transform(digits.target),
            learning_rate=1e-3, epochs=100,
            stratified=True
        )


class Iris(_AbstractDataset):
    def __init__(self):
        iris = datasets.load_iris()

        super().__init__(
            iris.data,
            LabelBinarizer().fit_transform(iris.target),
            stratified=True
        )


def _mulan(name):
    data_root = 'http://sourceforge.net/projects/mulan/files/datasets/'

    # prepear
    href = data_root + name + '.rar'
    data_dir = path.join(data_home, name)
    file_rar = path.join(data_dir, name + '.rar')
    file_arff = path.join(data_dir, name + '.arff')
    file_npz = path.join(data_dir, name + '.npz')

    # data have not been feched and processed
    if not path.exists(file_npz):
        os.makedirs(data_dir, exist_ok=True)

        # download file
        urllib.request.urlretrieve(href, file_rar)

        # extract arff file from rar archive
        with rarfile.RarFile(file_rar) as rf:
            rf.extract(name + '.arff', path=data_dir)

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

        # save inputs and targets numpy arrays
        np.savez(file_npz, inputs=inputs, targets=targets)

    # data have all ready been processed, just load it
    data = np.load(file_npz)

    return (data['inputs'], data['targets'])


class Scene(_AbstractDataset):
    def __init__(self):
        inputs, targets = _mulan('scene')
        super().__init__(inputs, targets, stratified=False)


class Emotions(_AbstractDataset):
    def __init__(self):
        inputs, targets = _mulan('emotions')
        super().__init__(inputs, targets, stratified=False)


class CAL500(_AbstractDataset):
    def __init__(self):
        inputs, targets = _mulan('CAL500')
        super().__init__(inputs, targets, stratified=False)

all_datasets = [MNIST, Iris, Scene, Emotions, CAL500]
