#################################
# Your name:
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

"""
Q4.1 skeleton.

Please use the provided functions signature for the SVM implementation.
Feel free to add functions and other code, and submit this file with the name svm.py
"""


# generate points in 2D
# return training_data, training_labels, validation_data, validation_labels
def get_points():
    X, y = make_blobs(n_samples=120, centers=2, random_state=0, cluster_std=0.88)
    return X[:80], y[:80], X[80:], y[80:]


def create_plot(X, y, clf):
    plt.clf()

    # plot the data points
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.PiYG)

    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0] - 2, xlim[1] + 2, 30)
    yy = np.linspace(ylim[0] - 2, ylim[1] + 2, 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])


def train_one_kernel(X_train, y_train, X_val, y_val, **kwargs):
    clf = svm.SVC(**kwargs)
    clf.fit(X_train, y_train)
    n_support = clf.n_support_

    title = ', '.join('{}: {}'.format(k, v) for k, v in kwargs.items())
    create_plot(X_val, y_val, clf)
    plt.title(title)
    plt.show()

    print('Classifier: {}, Number of support vectors: {}'.format(title, n_support))
    return n_support


def train_three_kernels(X_train, y_train, X_val, y_val):
    """
    Returns: np.ndarray of shape (3,2) :
                A two dimensional array of size 3 that contains the number of support vectors for each class(2) in the three kernels.
    """
    res = np.zeros((3, 2))
    res[0] = train_one_kernel(X_train, y_train, X_val, y_val, kernel='linear', C=1000)
    res[1] = train_one_kernel(X_train, y_train, X_val, y_val, kernel='poly', C=1000, degree=2)
    res[2] = train_one_kernel(X_train, y_train, X_val, y_val, kernel='rbf', C=1000)
    return res


def linear_accuracy_per_C(X_train, y_train, X_val, y_val):
    """
        Returns: np.ndarray of shape (11,) :
                    An array that contains the accuracy of the resulting model on the VALIDATION set.
    """
    # TODO: add your code here


def rbf_accuracy_per_gamma(X_train, y_train, X_val, y_val):
    """
        Returns: np.ndarray of shape (11,) :
                    An array that contains the accuracy of the resulting model on the VALIDATION set.
    """
    # TODO: add your code here


def main():
    x_train, y_train, x_val, y_val = get_points()
    train_three_kernels(x_train, y_train, x_val, y_val)


if __name__ == '__main__':
    main()

