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


def perceptron(data, labels):
    """
    :return: nd array of shape (data.shape[1],) or (data.shape[1],1) representing the perceptron classifier
    """
    data = sklearn.preprocessing.normalize(data)
    n = len(labels)
    w = np.zeros(n)
    for i in range(n):
        prediction = np.sign(np.dot(w[i], data[i]))
        print(f'np.dot(w[i], data[i])={np.dot(w[i], data[i])}, sign={prediction}')
        if prediction != labels[i][0]:
            w = w + labels[i][0] * data[i]

#################################


def main():
    n = 10
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()

    train_stacked = np.column_stack((train_data, train_labels))
    np.random.shuffle(train_stacked)
    train_data_n = [train_stacked[i][:-1] for i in range(n)]
    train_labels_n = [train_stacked[i][-1] for i in range(n)]

    perceptron(train_data_n, train_labels_n)


if __name__ == '__main__':
    main()
#################################
