from ..model import Model
import numpy as np
import os
import pickle
from abc import ABCMeta
import tensorflow as tf


class tfModel(Model):
    """
    Generic abstract class for a TF model that mimics the scikit-learn interface
    """

    __metaclass__ = ABCMeta

    def __init__(self, nb_classes=10):
        self.nb_classes = nb_classes
        self.input = tf.placeholder(tf.float32, shape=(None, 784))
        self.output = tf.placeholder(tf.float32, shape=(None, self.nb_classes))
        self.output_probs = self.model(self.input)
        self.eps = tf.Variable(0.3)
        self.adversarial_example_sym = fgsm(self.input, self.output_probs, eps=self.eps, clip_min=0.0, clip_max=1.0)
        self.sess = tf.Session()

    def model(self, input_ph):
        # TO BE OVERRIDEN
        return False

    def fit(self, X_train, y_train):
        y_train = np_labels_to_one_hot(y_train, 10)
        tf_model_train(self.sess, self.input, self.output, self.output_probs, X_train, y_train)
        return True

    def score(self, X_test, y_test):
        res = np.argmax(batch_eval(self.sess, [self.input], [self.output_probs], [X_test])[0], axis=1)
        return float(np.sum(res == y_test)) / len(y_test)

    def predict(self, input):
        return np.argmax(self.sess.run([self.output_probs], feed_dict={self.input: input, keras.backend.learning_phase(): 0})[0], axis=1)

    def adversarial_example(self, input, eps=None):
        if len(np.shape(input)) == 1:
            input = np.reshape(input, (1, 784))
        if eps is not None:
            self.eps.assign(float(eps))
        adversarial_example_val = self.sess.run([self.adversarial_example_sym], feed_dict={self.input: input, keras.backend.learning_phase(): 0})[0]

return adversarial_example_val