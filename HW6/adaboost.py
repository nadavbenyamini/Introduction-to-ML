#################################
# Your name:
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib.
from matplotlib import pyplot as plt
import numpy as np
from process_data import parse_data

np.random.seed(7)


def sign(x):
    return 1 if x >= 0 else -1


def update_D(D, ht, X_train, y_train, wt):
    n = len(y_train)
    sum_D = sum(D[k] * np.exp(wt * y_train[k] * ht(X_train, k)) for k in range(n))
    return [D[i] * np.exp(wt * y_train[i] * ht(X_train, i))/sum_D for i in range(n)]


def update_w(et):
    return np.log((1 - et) / et) / 2


def get_weak_learner(D, X_train, y_train):
    n = len(X_train)
    m = len(X_train[0])
    best_j = 0
    best_theta = 0
    best_direction = 1
    best_f = np.inf
    threshold = 0.005

    # Some randomness to make it interesting
    indices = list(range(m))
    np.random.shuffle(indices)

    # Finding the best theta for each j, then finding best j
    for j in indices:
        if best_f < threshold:  # To speed things up
            break
        for direction in [-1, 1]:
            if best_f < threshold:  # To speed things up
                break
            X_train_j = sorted(map(lambda x_i: x_i[j], X_train), reverse=direction < 0)
            X_train_j.append(X_train_j[-1] + 1)
            f = sum(D[i] * int(y_train[i] == 1) for i in range(n))
            if f < best_f:
                best_f = f
                best_theta = X_train_j[0] - 1
                best_j = j

            for i in range(n):
                if best_f < threshold:  # To speed things up
                    break
                f = f - y_train[i] * D[i]
                if f < best_f and i + 1 < n and X_train_j[i] != X_train_j[i + 1]:
                    best_f = f
                    best_theta = (X_train_j[i] + X_train_j[i + 1])/2
                    best_j = j
                    best_direction = direction
                    # print('Better hypothesis found: j={}, f={}, theta={}, direction={}'.format(best_j, best_f, best_theta, best_direction))

    best_h = lambda x, i: sign((best_theta - x[i][best_j]) * best_direction)
    print('Bye')
    return best_h


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
    n = len(X_train)
    D = [1 / n] * n
    alpha_vals = []
    hypotheses = []
    for t in range(T):
        ht = get_weak_learner(D, X_train, y_train)
        et = sum(D[i] * int(ht(X_train, i) != y_train[i]) for i in range(n))
        print('~~~ Iteration #{}, Error = {} ~~~'.format(t, et))
        if et <= 0 or et >= 1:
            continue
        wt = update_w(et)
        D = update_D(D, ht, X_train, y_train, wt)
        # print('~~~ Iteration #{}, D[:20] = {} ~~~'.format(t, D[:20]))
        hypotheses.append(ht)
        alpha_vals.append(wt)
    return hypotheses, alpha_vals


##############################################
# You can add more methods here, if needed.

##############################################


def main():
    print('Started')
    data = parse_data()
    if not data:
        return
    (X_train, y_train, X_test, y_test, vocab) = data

    T = 5
    hypotheses, alpha_vals = run_adaboost(X_train, y_train, T)

    train_errors = []
    testt_errors = []
    for t in range(T):
        train_errors.append(sum(alpha_vals[j] * hypotheses[j](X_train, i) for i in range(len(X_train)) for j in range(t)))
        testt_errors.append(sum(alpha_vals[j] * hypotheses[j](X_test, i) for i in range(len(X_test)) for j in range(t)))

    plt.plot(train_errors, color="blue", label='Train')
    plt.plot(testt_errors, color="red", label='Test')
    plt.xlabel('Iterations')
    plt.ylabel('error')
    plt.show()


if __name__ == '__main__':
    main()

