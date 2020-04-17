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

def get_n_samples(data, labels, n):
    stacked = np.column_stack((data, labels))
    np.random.shuffle(stacked)
    n_data = [stacked[i][:-1] for i in range(n)]
    n_labels = [stacked[i][-1] for i in range(n)]
    return n_data, n_labels


def sign(x):
    if x == 0:
        return 0
    return 1 if x > 0 else -1


def predict(w, x):
    dp = np.dot(w, x)
    return sign(dp.astype(float))


def perceptron(data, labels):
    """
    :return: nd array of shape (data.shape[1],) or (data.shape[1],1) representing the perceptron classifier
    """
    data = sklearn.preprocessing.normalize(data)
    w = np.zeros(data[0].shape[0])
    for i in range(len(labels)):
        if np.where(predict(w, data[i]) != labels[i]):
            w += np.dot(labels[i], data[i])
    return w


def calc_accuracy(w, data, labels):
    n = len(labels)
    return sum([predict(w, data[i]) == labels[i] for i in range(n)]) / n


def display_table(rows, columns):
    fig, ax = plt.subplots()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    plt.table(cellText=rows, colLabels=columns, loc='center')
    plt.show()  # TODO - Comment out


def main():
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()
    n_list = [5, 10, 50, 100, 500, 1000, 5000]
    accuracies = {}
    for n in n_list:
        accuracies[n] = []
        for j in range(10):
            train_data_n, train_labels_n = get_n_samples(train_data, train_labels, n)
            w = perceptron(train_data_n, train_labels_n)
            accuracies[n].append(calc_accuracy(w, test_data, test_labels))
        print('n = {}, mean_accuracy = {}, median_accuracy = {}'.format(n, np.mean(accuracies[n]), np.median(accuracies[n])))

    c = ['N', 'Mean Accuracy', '95% Percentile', '5% Percentile']
    r = []
    for n in accuracies:
        r += [[n, np.mean(accuracies[n]), np.percentile(accuracies[n], q=0.95),  np.percentile(accuracies[n], q=0.05)]]

    # Q1:
    display_table(r, c)

    # Q2:
    plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest')


if __name__ == '__main__':
    main()
#################################
