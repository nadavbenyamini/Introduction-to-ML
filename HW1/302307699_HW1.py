from sklearn.datasets import fetch_openml
from sklearn.metrics.pairwise import euclidean_distances
import numpy.random
import matplotlib.pyplot as plt
import operator

# List neighbours for each test image
CACHE = {}


def get_distance(train_img, test_image, train_index, test_index):
    try:
        distance = CACHE[train_index][test_index]
    except KeyError:
        distance = euclidean_distances(train_img, test_image)
        if train_index not in CACHE:
            CACHE[train_index] = {}
        CACHE[train_index][test_index] = distance
    return distance


def get_neighbours(train_img_data, train_img_labels, test_image, test_image_index):
    test_image = test_image.reshape(1, -1)
    neighbors = []
    for i in range(len(train_img_data)):
        train_img = train_img_data[i].reshape(1, -1)
        distance = get_distance(train_img, test_image, train_index=i, test_index=test_image_index)
        label = train_img_labels[i]
        neighbors.append((distance, label))
    return neighbors


def predict(train_img_data, train_img_labels, test_image, test_image_index, k):

    # Fetching k-nearest-neighbours
    neighbours = get_neighbours(train_img_data, train_img_labels, test_image, test_image_index)
    k_neighbours = sorted(neighbours)[:k]

    # Counting most popular label
    results = {str(i): 0 for i in range(10)}
    for n in k_neighbours:
        results[n[1]] += 1

    return max(results.items(), key=operator.itemgetter(1))[0]


def launch_model(train, train_labels, test, test_labels, n, k):
    total_loss = 0
    test_size = len(test)
    for i in range(test_size):
        result = predict(train_img_data=train[:n],
                         train_img_labels=train_labels,
                         test_image=test[i],
                         test_image_index=i,
                         k=k)
        total_loss += test_labels[i] != result
    return (test_size - total_loss) / test_size


def main():
    # Setup:
    mnist = fetch_openml('mnist_784')
    data = mnist['data']
    labels = mnist['target']

    idx = numpy.random.RandomState(0).choice(70000, 11000)
    train = data[idx[:10000], :].astype(int)
    train_labels = labels[idx[:10000]]
    test = data[idx[10000:], :].astype(int)
    test_labels = labels[idx[10000:]]
    
    # ------------------- Question B ------------------- #
    # Should be x. For a random classifier accuracy should be around 1/10 (uniform distribution between 10 digits)
    accuracy = launch_model(train, train_labels, test, test_labels, n=1000, k=10)
    print('Accuracy = {}'.format(str(accuracy)))
    # -------------------------------------------------- #

    # ------------------- Question C ------------------- #
    k_list = list(range(1, 101))
    accuracy_list = []
    for k in k_list:
        accuracy_list.append(launch_model(train, train_labels, test, test_labels, n=1000, k=k))

    plt.plot(k_list, accuracy_list)
    plt.savefig('hw1_graph_1.png')
    # -------------------------------------------------- #

    # ------------------- Question D ------------------- #
    n_list = [i*100 for i in range(1, 51)]
    accuracy_list2 = []
    for n in n_list:
        accuracy_list2.append(launch_model(train, train_labels, test, test_labels, n=n, k=1))

    plt.close()
    plt.plot(n_list, accuracy_list2)
    plt.savefig('hw1_graph_2.png')
    # -------------------------------------------------- #


if __name__ == '__main__':
    main()
