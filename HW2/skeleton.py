#################################
# Your name: Nadav Benyamini
#################################

import numpy as np
import matplotlib.pyplot as plt

# TODO - Restore imports
import intervals
from HW2.intervals import find_best_interval

A1 = [(0, 0.2), (0.4, 0.6), (0.8, 1)]
P1 = 0.8
P0 = 0.1
DELTA = 0.1


class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        x = list(np.random.random_sample(m))
        x.sort()
        res = np.zeros((m, 2))

        def p(xi):
            r = np.random.rand()
            for interval in A1:
                if interval[0] < xi < interval[1]:
                    return r < P1
            return r < P0

        for i in range(m):
            res[i] = [x[i], p(x[i])]

        return res

    def draw_sample_intervals(self, m, k):
        """
        Plots the data as asked in (a) i ii and iii.
        Input: m - an integer, the size of the data sample.
               k - an integer, the maximum number of intervals.

        Returns: None.
        """
        sample = self.sample_from_D(m)
        model = get_model(sample, k=k)

        plt.scatter(x=sample[:, 0], y=sample[:, 1])
        # adding vertical lines for intervals
        for i in model:
            plt.axvline(i[0], linestyle='--', color='k')
            plt.axvline(i[1], linestyle='--', color='k')
            plt.axvspan(i[0], i[1], alpha=0.25, color='red')

        plt.ylim(-0.1, 1.1)
        plt.xlim(0, 1)
        plt.yticks([-0.1 + 0.1 * i for i in range(13)])
        plt.gca().xaxis.grid(True)

        plt.show()  # TODO: COMMENT OUT
        plt.close()

    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        size = (m_last - m_first) // step
        res = np.zeros(shape=(size, 3))
        for i in range(size):
            m = m_first + i*step
            emp_errors, true_errors = [], []
            for t in range(T):
                sample = self.sample_from_D(m)
                model = get_model(sample, k=k)
                emp_errors.append(empirical_error(sample, model))
                true_errors.append(true_error(model))
            res[i] = [
                m,
                np.array(true_errors).mean(),
                np.array(emp_errors).mean()
            ]

        fig, ax1 = plt.subplots()
        ax1.plot(res[:, 0], res[:, 1], 'blue',
                 res[:, 0], res[:, 2], 'red')

        ax1.set_xlabel('m')
        ax1.set_ylabel('Error')
        ax1.legend(['True', 'Empirical'])

        plt.show()  # TODO: COMMENT OUT
        plt.close()
        return res

    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        size = 1 + (k_last - k_first) // step
        res = np.zeros(shape=(size, 3))
        best_error = 1000
        best_k = -1
        for i in range(size):
            k = k_first + i*step
            sample = self.sample_from_D(m)
            model = get_model(sample, k=k)
            emp = empirical_error(sample, model)
            if emp < best_error:
                best_error = emp
                best_k = k
            res[i] = [k, true_error(model), emp]

        fig, ax1 = plt.subplots()

        ax1.set_xlabel('k')
        ax1.plot(res[:, 0], res[:, 1], 'blue',
                 res[:, 0], res[:, 2], 'red')
        ax1.legend(['True Error (k)', 'Empirical Error'])

        plt.show()  # TODO: COMMENT OUT

        return best_k

    def experiment_k_range_srm(self, m, k_first, k_last, step):
        """Runs the experiment in (d).
        Plots additionally the penalty for the best ERM hypothesis.
        and the sum of penalty and empirical error.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the SRM algorithm.
        """
        size = 1 + (k_last - k_first) // step
        res = np.zeros(shape=(size, 5))
        best_error = 1000
        best_k = -1
        for i in range(size):
            k = k_first + i*step
            sample = self.sample_from_D(m)
            model = get_model(sample, k=k)
            emp = empirical_error(sample, model)
            penalty = get_penalty(m, k)
            if emp < best_error:
                best_error = emp
                best_k = k
            res[i] = [k, true_error(model), emp, penalty, emp + penalty]

        fig, ax1 = plt.subplots()

        ax1.set_xlabel('k')
        ax1.plot(res[:, 0], res[:, 1], 'blue',
                 res[:, 0], res[:, 2], 'red',
                 res[:, 0], res[:, 3], 'grey',
                 res[:, 0], res[:, 4], 'green')

        ax1.legend(['True Error (k)', 'Empirical Error', 'Penalty', 'Empirical Error + Penalty'])

        plt.show()  # TODO: COMMENT OUT
        return best_k

    def cross_validation(self, m, T):
        """Finds a k that gives a good test error.
        Chooses the best hypothesis based on 3 experiments.
        Input: m - an integer, the size of the data sample.
               T - an integer, the number of times the experiment is performed.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        sample = self.sample_from_D(m)
        indices = [i for i in range(m)]
        best_k_stats = {}
        for i in range(T):
            best_res = 10000
            best_k = 0
            np.random.shuffle(indices)
            test_indices = sorted(indices[:m//5])
            test_sample = np.array([(sample[i][0], sample[i][1]) for i in test_indices])
            validation_indices = sorted(indices[m//5:])
            validation_sample = np.array([(sample[i][0], sample[i][1]) for i in validation_indices])
            for k in range(1, 10):
                model = get_model(test_sample, k=k)
                emp = empirical_error(validation_sample, model)
                penalty = get_penalty(m, k)
                res = emp + penalty
                if res <= best_res:
                    best_res = res
                    best_k = k
            print('Experiment {}, best k found: {}'.format(i, best_k))
            if best_k in best_k_stats:
                best_k_stats[best_k] += 1
            else:
                best_k_stats[best_k] = 1
        return sorted(best_k_stats.items(), key=lambda stats: -1*stats[1])[0][0]

#################################
# Place for additional methods


def get_model(sample, k):
    return find_best_interval(xs=sample[:, 0], ys=sample[:, 1], k=k)[0]


def empirical_error(sample, model):
    m = len(sample[:, 0])
    predictions = predict(sample, model)
    sample_y = sample[:, 1]
    error = 0
    for i in range(m):
        error += float(sample_y[i]) != float(predictions[i])
    return error / m


def true_error(model):
    reverse_model = model_compliment(model)
    A0 = model_compliment(A1)
    h0a0 = intervals_intersections(reverse_model, A0)
    h0a1 = intervals_intersections(reverse_model, A1)
    h1a0 = intervals_intersections(model, A0)
    h1a1 = intervals_intersections(model, A1)

    return sum([
        P0 * sum([i[1] - i[0] for i in h0a0]),
        (1-P0) * sum([i[1] - i[0] for i in h1a0]),
        P1 * sum([i[1] - i[0] for i in h0a1]),
        (1-P1) * sum([i[1] - i[0] for i in h1a1])
    ])


def intervals_intersections(l1, l2):
    """
    :param l1: list of intervals
    :param l2: list of intervals
    :return: List of intersecting intervals
    Assumption: both l1 and l2 are sorted and don't have overlapping intervals
    """
    res = []
    for i1 in l1:
        for i2 in l2:
            if i2[0] > i1[1] or i2[1] < i1[0]:
                continue
            res.append((max(i2[0], i1[0]), min(i2[1], i1[1])))
    return res


# Returns complementing intervals
def model_compliment(model):
    """
    :param model: list of intervals
    :return: complementing intervals

    for example if model = [(0.3, 0.5)] then model_compliment = [(0, 0.3), (0.5, 1)]
    """
    padded_model = [(-1, 0)] + model + [(1, 2)]
    return [(padded_model[i][1], padded_model[i+1][0]) for i in range(len(model)+1)]


def predict(sample, model):
    sample_x = [float(x) for x in sample[:, 0]]
    m = len(sample_x)
    predictions = [0] * m
    for i in range(m):
        for interval in model:
            if interval[0] < sample_x[i] < interval[1]:
                predictions[i] = 1
    return predictions


def get_penalty(m, k):
    vc_dim = 2*k  # From theoretical questions
    return np.sqrt((1/m) * (vc_dim*np.log(m/vc_dim) + np.log(1/DELTA)))
#################################


if __name__ == '__main__':
    ass = Assignment2()
    ass.draw_sample_intervals(100, 3)
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.experiment_k_range_srm(1500, 1, 10, 1)
    ass.cross_validation(1500, 3)

