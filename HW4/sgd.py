#################################
# Your name: Nadav Benyamini
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib

import numpy as np
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing
import matplotlib.pyplot as plt
import operator
import warnings
warnings.filterwarnings("error")


"""
Assignment 4 question 2 skeleton.

Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""


def helper_hinge():
    mnist = fetch_openml('mnist_784')
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def helper_ce():
    mnist = fetch_openml('mnist_784')
    data = mnist['data']
    labels = mnist['target']

    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:8000] != 'a'))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[8000:10000] != 'a'))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = labels[train_idx[:6000]]

    validation_data_unscaled = data[train_idx[6000:8000], :].astype(float)
    validation_labels = labels[train_idx[6000:8000]]

    test_data_unscaled = data[8000 + test_idx, :].astype(float)
    test_labels = labels[8000 + test_idx]

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def SGD_hinge(data, labels, C, eta_0, T):
    """
    Implements Hinge loss using SGD.
    """
    data = sklearn.preprocessing.normalize(data)
    w = np.zeros(data[0].shape[0])
    n = len(labels)
    for t in range(T):
        rand = np.random.randint(0, n)
        eta = eta_0 / (t+1)
        if np.dot(w*labels[rand], data[rand]) < 1:
            gradient = -np.dot(labels[rand], data[rand])
            w = (1-eta)*w - eta*C*gradient
        else:
            w = (1-eta)*w
    return w


def SGD_ce(data, labels, eta_0, T):
    """
    Implements multi-class cross entropy loss using SGD.
    """
    data = sklearn.preprocessing.normalize(data)
    L = 10  # 10 Possible labels
    w_arr = [np.zeros(data[0].shape[0])] * L
    n = len(labels)
    for t in range(T):
        rand = np.random.randint(0, n)
        sum_exp = sum(np.exp(np.dot(w, data[rand])) for w in w_arr)
        eta = eta_0 / (t+1)
        for i in range(L):
            gradient = ce_gradient(i, w_arr[i], data[rand], labels[rand], sum_exp)
            w_arr[i] = w_arr[i] - eta*gradient
    return w_arr


def ce_gradient(i, w, x, y, sum_exp):
    indicator = int(str(i) == str(y))
    p = (np.exp(np.dot(w, x))) / sum_exp
    gradient = (p - indicator) * x
    return gradient

#################################


def predict(w, x):
    dp = np.dot(w, x)
    return 1 if dp.astype(float) >= 0 else -1


def calc_accuracy(w, data, labels):
    n = len(labels)
    return sum(predict(w, data[i]) == labels[i] for i in range(n)) / n


def calc_accuracy_multi_labels(w_arr, data, labels):
    n = len(labels)
    return sum(get_max_prediction(w_arr, data[i]) == labels[i] for i in range(n)) / n


def get_max_prediction(w_arr, x):
    all_predictions = {str(i): np.dot(w, x) for i, w in enumerate(w_arr)}
    # print(all_predictions)
    return max(all_predictions.items(), key=operator.itemgetter(1))[0]


def q1_main():
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper_hinge()
    accuracies = {}

    # Q1a
    for k in range(-5, 6):
        eta_0 = 10**k
        accuracies[eta_0] = []
        try:
            for j in range(10):
                sgd = SGD_hinge(data=train_data, labels=train_labels, C=1, eta_0=eta_0, T=1000)
                accuracies[eta_0].append(calc_accuracy(w=sgd, data=validation_data, labels=validation_labels))
            accuracies[eta_0] = sum(accuracies[eta_0]) / 10
        except RuntimeWarning:
            del accuracies[eta_0]
            continue
    plt.plot(list(accuracies.keys()), list(accuracies.values()), '-o')
    plt.xscale('log')
    plt.xlabel('eta_0')
    plt.ylabel('Average Accuracy')
    plt.show()  # TODO - Comment out

    # Q1b
    best_eta = max(accuracies.items(), key=operator.itemgetter(1))[0]
    print('Best eta0 = {}'.format(best_eta))
    accuracies = {}
    for k in range(-5, 6):
        c = 10**k
        accuracies[c] = []
        try:
            for j in range(10):
                sgd = SGD_hinge(data=train_data, labels=train_labels, C=c, eta_0=best_eta, T=1000)
                accuracies[c].append(calc_accuracy(w=sgd, data=validation_data, labels=validation_labels))
            accuracies[c] = sum(accuracies[c]) / 10
        except RuntimeWarning:
            del accuracies[c]
            continue
    plt.plot(list(accuracies.keys()), list(accuracies.values()), '-o')
    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('Average Accuracy')
    plt.show()  # TODO - Comment out

    # Q1c
    best_c = max(accuracies.items(), key=operator.itemgetter(1))[0]
    print('Best c = {}'.format(best_c))
    sgd = SGD_hinge(data=train_data, labels=train_labels, C=best_c, eta_0=best_eta, T=2000)
    plt.imshow(np.reshape(sgd, (28, 28)), interpolation='nearest')
    plt.show()  # TODO - Comment out

    # Q1d
    print('Accuracy on test data = {}'.format(calc_accuracy(sgd, test_data, test_labels)))


def q2_main():
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper_ce()
    accuracies = {}

    # Q2a
    for k in range(-5, 6):
        eta_0 = 10 ** k
        accuracies[eta_0] = []
        try:
            for j in range(10):
                sgd = SGD_ce(data=train_data, labels=train_labels, eta_0=eta_0, T=1000)
                accuracies[eta_0].append(calc_accuracy_multi_labels(sgd, validation_data, validation_labels))
            accuracies[eta_0] = sum(accuracies[eta_0]) / 10
        except RuntimeWarning:
            del accuracies[eta_0]
            continue
    plt.plot(list(accuracies.keys()), list(accuracies.values()), '-o')
    plt.xscale('log')
    plt.xlabel('eta_0')
    plt.ylabel('Average Accuracy')
    plt.show()  # TODO - Comment out

    # Q2a
    best_eta = max(accuracies.items(), key=operator.itemgetter(1))[0]
    print('Best eta = {}'.format(best_eta))
    sgd = SGD_ce(data=train_data, labels=train_labels, eta_0=best_eta, T=2000)
    for i, w in enumerate(sgd):
        plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest')
        plt.title('i={}'.format(i))
        plt.show()  # TODO - Comment out

    # Q3a
    print('Accuracy on test data = {}'.format(calc_accuracy_multi_labels(sgd, test_data, test_labels)))


def main():
    print('\nStarting Q1:')
    q1_main()
    print('\nStarting Q2:')
    q2_main()


if __name__ == '__main__':
    main()
