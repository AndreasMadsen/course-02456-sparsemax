
import numpy as np
from sklearn.model_selection import StratifiedKFold


class ModelEvaluator:
    def __init__(self, model, dataset, epochs=1000,
                 random_state=None, verbose=False):
        self.model = model

        self.epochs = epochs
        self.random_state = random_state
        self.verbose = verbose

        # setup data
        self.x = dataset.inputs
        self.t = dataset.targets

    def single_fold(self, fold, train_index, test_index):
        # select datasets
        x_train = self.x[train_index, :]
        t_train = self.t[train_index, :]
        x_test = self.x[test_index, :]
        t_test = self.t[test_index, :]

        # reset model
        self.model.reset()

        # train model
        self.model.update(x_train, t_train, epochs=self.epochs)

        # prediction on test data
        target = np.argmax(t_test, axis=1)
        predict = np.argmax(self.model.predict(x_test), axis=1)
        missrate = np.mean(predict != target)

        # TODO: change this to a KL distance when a target distribution is used
        return missrate

    def all_folds(self, n_splits=5):
        # cross validation
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self.random_state
        )
        splits = skf.split(self.x, np.argmax(self.t, axis=1))

        # collect missrate data
        missrates = np.zeros(n_splits)
        for fold, (train_index, test_index) in enumerate(splits):
            # fit model and get final test performance
            missrates[fold] = self.single_fold(fold, train_index, test_index)

        return missrates
