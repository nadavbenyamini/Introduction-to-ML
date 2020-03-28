#################################
# Your name: Nadav Benyamini
#################################

import numpy as np
import matplotlib.pyplot as plt

# TODO - Restore imports
import intervals
from HW2.intervals import find_best_interval


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

        def p(x_i):
            r = np.random.rand()
            if (0 < x_i < 0.2) or (0.4 < x_i < 0.6) or (0.8 < x_i < 1):
                return r < 0.8
            return r < 0.1

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
        model = self.get_model(sample, k=k)

        # Note: I chose to plot the intervals as colors
        # red dot = x is in one of the bets_intervals
        colors = ['blue']*m
        for i in range(m):
            xi = float(sample[:, 0][i])
            for interval in model:
                if interval[0] < xi < interval[1]:
                    colors[i] = 'red'

        plt.scatter(x=sample[:, 0], y=sample[:, 1], c=colors)
        plt.ylim(-0.1, 1.1)
        plt.xlim(0, 1)
        plt.yticks([-0.1 + 0.1 * i for i in range(13)])
        plt.gca().xaxis.grid(True)

        # plt.show()  # TODO: REMOVE

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
            for t in range(10):
                sample = self.sample_from_D(m)
                model = self.get_model(sample, k=k)
                emp_errors.append(self.empirical_error(sample, model))
                true_errors.append(self.true_error(model))

            res[i] = [m, np.array(emp_errors).mean(), np.array(true_errors).mean()]

        print(res)
        plt.close()
        plt.scatter(x=res[:, 0], y=res[:, 1])
        plt.show()
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
        # TODO: Implement the loop
        pass

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
        # TODO: Implement the loop
        pass

    def cross_validation(self, m, T):
        """Finds a k that gives a good test error.
        Chooses the best hypothesis based on 3 experiments.
        Input: m - an integer, the size of the data sample.
               T - an integer, the number of times the experiment is performed.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        # TODO: Implement me
        pass

    #################################
    # Place for additional methods
    def get_model(self, sample, k):
        return find_best_interval(xs=sample[:, 0], ys=sample[:, 1], k=k)[0]

    def empirical_error(self, sample, model):
        return 0.5

    def true_error(self, model):
        return 0.3

    #################################


if __name__ == '__main__':
    ass = Assignment2()
    ass.draw_sample_intervals(100, 3)
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.experiment_k_range_srm(1500, 1, 10, 1)
    ass.cross_validation(1500, 3)

