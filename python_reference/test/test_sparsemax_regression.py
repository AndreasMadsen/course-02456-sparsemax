import _test
from nose.tools import *

import collections

import numpy as np
from sklearn import datasets
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import StratifiedKFold

from sparsemax_regression import SparsemaxRegression

ModelResults = collections.namedtuple('ModelResults', ['loss', 'missrate'])


class ModelTest:
    def __init__(self, model, dataset, epochs=1000,
                 random_state=None, verbose=False):
        self.model = model
        self.dataset = dataset

        self.epochs = epochs
        self.random_state = random_state
        self.verbose = verbose

        # setup data
        self.x = dataset.data
        self.t = LabelBinarizer().fit_transform(dataset.target)

    def test_fold(self, fold, train_index, test_index):
        # select datasets
        x_train = self.x[train_index, :]
        t_train = self.t[train_index, :]
        x_test = self.x[test_index, :]
        t_test = self.t[test_index, :]

        # reset model
        self.model.reset()

        # train model
        for epoch in range(0, self.epochs):
            self.model.update(x_train, t_train)

        # prediction on test data
        loss = self.model.loss(x_test, t_test)
        target = np.argmax(t_test, axis=1)
        predict = np.argmax(self.model.predict(x_test), axis=1)
        missrate = np.mean(predict != target)

        # verbose output
        if (self.verbose):
            print('cross validation fold: %d' % fold)
            print('      loss: %f' % loss)
            print('  missrate: %f' % missrate)
            print('   predict: %s' % np.array_str(predict))
            print('    target: %s' % np.array_str(target))

        return ModelResults(loss, missrate)

    def test(self, loss, missrate):
        # cross validation
        skf = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=self.random_state
        )

        for fold, (train_index, test_index) in \
                enumerate(skf.split(self.dataset.data, self.dataset.target)):
            # fit model and get final test performance
            results = self.test_fold(fold, train_index, test_index)

            # assert prediction scores
            assert_true(results.loss < loss)
            assert_true(results.missrate < missrate)


def test_iris_classifier():
    """check sparsemax regression on the iris dataset"""

    # get data
    iris = datasets.load_iris()

    # intialize model
    model = SparsemaxRegression(
        input_size=4, output_size=3,
        regualizer=1e-1, learning_rate=1e-2,
        random_state=42
    )

    # setup model tester
    tester = ModelTest(model, iris, random_state=42, verbose=False)
    tester.test(loss=0.2, missrate=0.1)


def test_mnist_classifier():
    """check sparsemax regression on the digits dataset"""

    # get data
    mnist = datasets.load_digits()

    # intialize model
    model = SparsemaxRegression(
        input_size=mnist.data.shape[1], output_size=10,
        regualizer=1e-1, learning_rate=1e-2,
        random_state=42
    )

    # setup model tester
    tester = ModelTest(model, mnist, random_state=42, verbose=False)
    tester.test(loss=0.2, missrate=0.1)
