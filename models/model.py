from abc import ABCMeta, abstractmethod


class Model:
    """
    Generic abstract class for a model. This class
    is designed to be inhereted by skModel and
    tfModel.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, X_train, y_train):
        """
        This function should train the model if a
        backup cannot be found. If it does train
        the model, it should save a backup.
        :param X_train:
        :param y_train:
        :return:
        """
        pass

    @abstractmethod
    def score(self, X_test, y_test):
        """
        This function should return the model's accuracy
        averaged over the input data given the expected label.
        :param X_test:
        :param y_test:
        :return:
        """
        pass

    @abstractmethod
    def predict(self, inputs, rtn_label=True):
        """
        This function should return the label or
        probability vector of a set of inputs.
        :param inputs:
        :param rtn_label: if boolean set to True, will return argmax
        :return:
        """
        pass

    @abstractmethod
    def adversarial_example(self, inputs):
        """
        This function will return the adversarial examples
        corresponding to a given set of inputs.
        :param inputs:
        :return:
        """
        pass
