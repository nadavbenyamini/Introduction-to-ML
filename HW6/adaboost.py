#################################
# Your name:
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib.
from matplotlib import pyplot as plt
import numpy as np
from process_data import parse_data

np.random.seed(7)

HYPOTHESES, ALPHA_VALS = [], []


def sign(x):
    return 1 if x >= 0 else -1


# h is a tuple of theta and j, this is an activation of the implied classifier on a sample x[i]
def activate_h(h, x, i):
    theta, j, direction = h
    return sign(theta - x[i][j]) * direction


def update_D(D, ht, X_train, y_train, wt):
    n = len(y_train)
    sum_D = sum(D[i] * np.exp(-wt * y_train[i] * activate_h(ht, X_train, i)) for i in range(n))
    return [(D[i] * np.exp(-wt * y_train[i] * activate_h(ht, X_train, i)))/sum_D for i in range(n)]


def update_e(D, ht, X_train, y_train):
    n = len(y_train)
    return sum(D[i] * int(activate_h(ht, X_train, i) != y_train[i]) for i in range(n))


def update_w(et):
    return np.log((1 / et) - 1) / 2


def get_weak_learner(D, X_train, y_train):
    n = len(X_train)
    m = len(X_train[0])
    best_j = 0
    best_theta = 0
    best_direction = 1
    best_f = np.inf

    # Finding the best theta for each j, then finding best j
    for direction in [1, -1]:
        for j in range(m):

            # Sorting x, y and D together by x[j]
            train_tuples = sorted([(X_train[i][j], y_train[i], D[i]) for i in range(n)],
                                  key=lambda tup: (tup[0], np.random.rand()))
            X_train_j = [tup[0] for tup in train_tuples]
            y_train_j = [tup[1] for tup in train_tuples]
            D_j = [tup[2] for tup in train_tuples]

            X_train_j.append(X_train_j[-1] + 1)

            f = sum(D_j[i] * int(y_train_j[i] == float(direction)) for i in range(n))

            if f < best_f:
                best_f = f
                best_theta = X_train_j[0] - 1
                best_j = j

            for i in range(n):
                f = f - direction*y_train_j[i]*D_j[i]
                if f < best_f and X_train_j[i] != X_train_j[i + 1]:
                    best_f = f
                    best_theta = (X_train_j[i] + X_train_j[i + 1])/2
                    best_j = j
                    best_direction = direction
                    # print('Best h found: j={}, f={}, theta={}, direction={}'.format(best_j, best_f, best_theta, best_direction))

    return best_theta, best_j, best_direction


def run_adaboost(X_train, y_train, T):
    """
    Returns: 
        hypotheses:
            A list of T tuples describing the hypotheses chosen by the algorithm. 
            Each tuple has 3 elements (h_pred, h_index, h_theta), where h_pred is 
            the returned value (+1 or -1) if the count at index h_index is <= h_theta.

        alpha_vals:
            A list of T float values, which are the alpha values obtained in every 
            iteration of the algorithm.
    """
    if HYPOTHESES and ALPHA_VALS:
        print('Taking from Cache')
        return HYPOTHESES, ALPHA_VALS

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

        HYPOTHESES.append(ht)
        ALPHA_VALS.append(wt)

    return hypotheses, alpha_vals


def question1(data):
    print('Started Q1')

    def loss(X, y, hypotheses, alpha_vals):
        n = len(y)
        t = len(hypotheses)
        pred = [sign(sum(alpha_vals[k] * (activate_h(hypotheses[k], X, i)) for k in range(t))) for i in range(n)]
        return sum(pred[i] != y[i] for i in range(n)) / n
    questions1_3(data, loss, 'Accuracy')


def question2(data):
    print('Started Q2')
    T = 10
    (X_train, y_train, X_testt, y_testt, vocab) = data
    hypotheses, alpha_vals = run_adaboost(X_train, y_train, T)

    for t, h in enumerate(hypotheses):
        theta, j, direction = h
        print('Iteration: #{}, Dimension: {}, Theta: {}, Direction: {}, Word: "{}"'
              .format(t, j, theta + 1, direction, vocab[j]))


def question3(data):
    print('Started Q3')

    def loss(X, y, hypotheses, alpha_vals):
        n = len(y)
        t = len(hypotheses)
        return sum([np.exp(-y[i] * sum([alpha_vals[k] * activate_h(hypotheses[k], X, i) for k in range(t)])) for i in
                    range(n)]) / n

    questions1_3(data, loss, 'AVG Exponential Loss')


def questions1_3(data, loss, title):
    T = 80
    (X_train, y_train, X_testt, y_testt, vocab) = data
    hypotheses, alpha_vals = run_adaboost(X_train, y_train, T)

    train_errors = []
    testt_errors = []

    print('\nSummary:')
    for t in range(len(hypotheses)):
        hypos = hypotheses[:t]
        alphas = alpha_vals[:t]
        train_errors.append(loss(X_train, y_train, hypos, alphas))
        testt_errors.append(loss(X_testt, y_testt, hypos, alphas))

    plt.plot(train_errors, color="blue", label='Train', marker='o')
    plt.plot(testt_errors, color="red", label='Test', marker='o')
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel(title)
    plt.show()


def main():
    print('Started')
    data = parse_data()
    if not data:
        return
    question1(data)
    question2(data)
    question3(data)


if __name__ == '__main__':
    main()

