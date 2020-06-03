#################################
# Your name:
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib.
from matplotlib import pyplot as plt
import numpy as np
from process_data import parse_data

np.random.seed(7)

MARKED_DIMS = set()


def sign(x):
    return 1 if x >= 0 else -1


def update_D(D, ht, X_train, y_train, wt):
    n = len(y_train)
    sum_D = sum(D[i] * np.exp(-wt * y_train[i] * ht(X_train, i)) for i in range(n))
    return [(D[i] * np.exp(-wt * y_train[i] * ht(X_train, i)))/sum_D for i in range(n)]


def update_e(D, ht, X_train, y_train):
    n = len(y_train)
    return sum(D[i] * int(ht(X_train, i) != y_train[i]) for i in range(n))


def update_w(et):
    return np.log((1 / et) - 1) / 2


def get_weak_learner(D, X_train, y_train):
    n = len(X_train)
    m = len(X_train[0])
    best_j = 0
    best_theta = 0
    best_direction = 1
    best_f = np.inf
    threshold = 0.01

    # Some randomness to make it interesting
    indices = list(range(m))
    np.random.shuffle(indices)

    # Finding the best theta for each j, then finding best j
    for j in indices:

        # if j in MARKED_DIMS:  # Don't go for the same j twice
        #     continue

        # Sorting x and y together by x[j]
        train_tuples = sorted([(X_train[i][j], y_train[i], D[i]) for i in range(n)], key=lambda tup: (tup[0], np.random.rand()))
        X_train_j = [tup[0] for tup in train_tuples]
        y_train_j = [tup[1] for tup in train_tuples]
        D_j = [tup[2] for tup in train_tuples]

        X_train_j.append(X_train_j[-1] + 1)
        f = sum(D_j[i] * int(y_train_j[i] == 1.0) for i in range(n))

        if f < best_f:
            best_f = f
            best_theta = X_train_j[0] - 1
            best_j = j

        for i in range(n):
            direction = 1
            f = f - y_train_j[i]*D_j[i]
            # print(j, f, i, y_train_j[i], D_j[i])
            if f < 0:
                print(j, f, i, y_train_j[i], D_j[i], f + y_train_j[i]*D_j[i])
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                return
            if f < best_f and X_train_j[i] != X_train_j[i + 1]:
                best_f = f
                best_theta = (X_train_j[i] + X_train_j[i + 1])/2
                best_j = j
                best_direction = direction
                print('Best h found: j={}, f={}, theta={}, direction={}'.format(best_j, best_f, best_theta, best_direction))

    MARKED_DIMS.add(best_j)
    return lambda x, i: sign((best_theta - x[i][best_j])) * best_direction


def run_adaboost(X_train, y_train, T):
    """
    Returns: 
        hypotheses :
            A list of T tuples describing the hypotheses chosen by the algorithm. 
            Each tuple has 3 elements (h_pred, h_index, h_theta), where h_pred is 
            the returned value (+1 or -1) if the count at index h_index is <= h_theta.

        alpha_vals : 
            A list of T float values, which are the alpha values obtained in every 
            iteration of the algorithm.
    """
    n = len(y_train)
    D = [1 / n] * n
    et = 0
    wt = 0
    alpha_vals = []
    hypotheses = []
    for t in range(T):
        print('~~~ Starting iteration #{}, Error={}, wt={}, Top Di={} ~~~'
              .format(t, round(et, 5), round(wt, 5), [round(x, 5) for x in sorted(D)[:20]]))

        ht = get_weak_learner(D, X_train, y_train)
        et = update_e(D, ht, X_train, y_train)
        wt = update_w(et)
        D = update_D(D, ht, X_train, y_train, wt)
        hypotheses.append(ht)
        alpha_vals.append(wt)
    return hypotheses, alpha_vals


def main():
    print('Started')
    data = parse_data()
    if not data:
        return
    (X_train, y_train, X_testt, y_testt, vocab) = data
    T = 20
    hypotheses, alpha_vals = run_adaboost(X_train, y_train, T)

    train_errors = []
    testt_errors = []

    n_testt = len(y_testt)
    n_train = len(y_train)

    print('\nSummary:')
    for t in range(len(hypotheses)):
        train_pred = [sign(sum(alpha_vals[j] * (hypotheses[j](X_train, i)) for j in range(t))) for i in range(n_train)]
        testt_pred = [sign(sum(alpha_vals[j] * (hypotheses[j](X_testt, i)) for j in range(t))) for i in range(n_testt)]

        train_errors.append(sum(train_pred[i] != y_train[i] for i in range(n_train)) / n_train)
        testt_errors.append(sum(testt_pred[i] != y_testt[i] for i in range(n_testt)) / n_testt)

        print('~~~ Iteration #{}, Train Error = {}, Test Error = {} ~~~'.format(t, train_errors[-1], testt_errors[-1]))

    plt.plot(train_errors, color="blue", label='Train')
    plt.plot(testt_errors, color="red", label='Test')
    plt.legend()
    plt.xlabel('Iterations')
    plt.xticks(list(round(x) for x in range(len(hypotheses))))
    plt.ylabel('Error')
    plt.show()


if __name__ == '__main__':
    main()

