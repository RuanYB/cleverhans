
from ..model import Model
import numpy as np
import os
import pickle
from abc import ABCMeta, abstractmethod


class SKModel(Model):
    """
    Generic abstract class for a scikit-learn model that augments the
    scikit-learn interface
    """

    __metaclass__ = ABCMeta

    def __init__(self, save_path="/tmp/model"):
        self.model = None
        self.save_path = save_path

    def fit(self, X_train, y_train):
        if os.path.isfile(self.save_path):
            self.load()
        else:
            self.model.fit(X_train, y_train)
            self.save()
        return True

    def score(self, X_test, y_test):
        return self.model.score(X_test, y_test)

    def predict(self, X_test):
        if len(np.shape(X_test)) == 1:
            X_test = np.reshape(X_test, (1, int(np.shape(X_test)[0])))
        return self.model.predict(X_test)

    @abstractmethod
    def adversarial_example(self, input):
        pass

    def save(self):
        with open(self.save_path, 'wb') as pickle_file:
            pickle.dump(self.model, pickle_file)

    def load(self):
        with open(self.save_path, 'r') as pickle_file:
            self.model = pickle.load(pickle_file)