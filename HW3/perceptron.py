#################################
# Your name: Nadav Benyamini
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib

import numpy as np
import matplotlib.pyplot as plt
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing


"""
Assignment 3 question 1 skeleton.

Please use the provided function signature for the perceptron implementation.
Feel free to add functions and other code, and submit this file with the name perceptron.py
"""


def helper():
    mnist = fetch_openml('mnist_784')
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos)*2-1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos)*2-1

    test_data_unscaled = data[60000+test_idx, :].astype(float)
    test_labels = (labels[60000+test_idx] == pos)*2-1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


#################################
def sign(x):
    if x == 0:
        return 0
    return 1 if x > 0 else -1


def perceptron(data, labels):
    """
    :return: nd array of shape (data.shape[1],) or (data.shape[1],1) representing the perceptron classifier
    """
    data = sklearn.preprocessing.normalize(data)
    n = len(labels)
    m = data[0].shape[0]
    w = np.zeros(m)

    for i in range(n):
        dp = np.dot(w, data[i])
        prediction = sign(dp.astype(float))
        if np.where(prediction != labels[i]):
            w += np.dot(labels[i], data[i])
    return w


def calc_accuracy(w, data, labels):
    return 1


def get_n_data_and_labels(data, labels, n):
    stacked = np.column_stack((data, labels))
    np.random.shuffle(stacked)
    n_data = [data[i][:-1] for i in range(n)]
    n_labels = [labels[i][-1] for i in range(n)]
    return n_data, n_labels


def main():
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()

    for n in [5, 10]:
        train_data_n, train_labels_n = get_n_data_and_labels(train_data, train_labels, n)
        test_data_n, test_labels_n = get_n_data_and_labels(test_data, test_labels, n)
        w = perceptron(train_data_n, train_labels_n)
        accuracy = calc_accuracy(w, test_data_n, test_labels_n)
        print('n = {}, accuracy = {}'.format(n, accuracy))


if __name__ == '__main__':
    main()
#################################
