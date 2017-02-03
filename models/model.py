
from abc import ABCMeta, abstractmethod


class Model:
    """
    Generic abstract class for a model
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, X_train, y_train):
        pass

    @abstractmethod
    def score(self, X_test, y_test):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass

    @abstractmethod
    def adversarial_example(self, input):
        pass
