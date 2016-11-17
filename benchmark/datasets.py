
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
        digits = datasets.fetch_mldata('MNIST original', data_home=data_home)

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


class Scene(_AbstractDataset):
    def __init__(self):
        scene_dir = path.join(data_home, 'scene')
        href = 'http://sourceforge.net/projects/mulan/files/datasets/scene.rar'

        # data have not been feched and processed
        if not path.exists(path.join(scene_dir, 'scene.npz')):
            os.makedirs(scene_dir, exist_ok=True)

            # download file
            urllib.request.urlretrieve(href, path.join(scene_dir, 'scene.rar'))

            # extract arff file from rar archive
            with rarfile.RarFile(path.join(scene_dir, 'scene.rar')) as rf:
                rf.extract('scene.arff', path=scene_dir)

            # load arff file
            with open(path.join(scene_dir, 'scene.arff'), 'r') as f:
                data, meta = scipy.io.arff.loadarff(f)

            # get target and inputs names
            target_names = [
                'Beach', 'Sunset', 'FallFoliage', 'Field', 'Mountain', 'Urban'
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
            np.savez(
                path.join(scene_dir, 'scene.npz'),
                inputs=inputs, targets=targets
            )

        # data have all ready been processed, just load it
        data = np.load(
            path.join(scene_dir, 'scene.npz')
        )

        super().__init__(data['inputs'], data['targets'])

all_datasets = [Digits, Iris, Scene]
