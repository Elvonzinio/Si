import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from perceptron import SimplePerceptron


def data_cw4(size):
    X, y = make_blobs(n_samples=size, centers=2, n_features=2, cluster_std=0.5, random_state=0)

    theta = 1
    rotate_matrix = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    X = np.matmul(X, rotate_matrix)

    for i, element in enumerate(y):
        if element == 0:
            y[i] = -1
    y = y.reshape((size, 1))

    return X, y.reshape(1000)


def data_cw5(I):
    np.random.seed(0)
    X = np.random.uniform(0, 2 * np.pi, I)
    Y = np.random.uniform(-1, 1, I)
    x = np.c_[X, Y]
    y = np.ones(I)

    for i in range(I):
        if np.abs(np.sin(x[i, 0])) > np.abs(x[i, 1]):
            y[i] = -1

    for i in range(I):
        x[i, 0] = (x[i, 0] / np.pi) - 1

    return x, y


def cw4():
    X, y = data_cw4(1000)
    clf = SimplePerceptron(learningRate=1.0)
    clf.fit(X, y)
    print(clf.weights)

    triangles = []
    circles = []

    for element, label in zip(X, y):
        if label == 1:
            triangles.append(element)
        else:
            circles.append(element)

    triangles = np.array(triangles)
    circles = np.array(circles)

    plt.scatter(triangles[:, 0], triangles[:, 1], c='red', s=6, marker="^")
    plt.scatter(circles[:, 0], circles[:, 1], c='blue', s=6, marker="o")

    min_value = min(X[:, 0])
    max_value = max(X[:, 0])

    x1 = np.array([min_value, max_value])
    w = clf.weights
    x2 = (-(w[0] + w[1] * x1) / w[2])
    plt.plot(x1, x2)
    plt.show()


def cw5():
    clf = SimplePerceptron(learningRate=1.0, m=100)
    X, y = data_cw5(1000)
    c = clf.centers()
    z = clf.distance(X, c, sig=0.2)
    clf.fit(z, y, maxIteration=2000)

    print(clf.score(z, y))

    labels, XX, YY = clf.contour(c, 100)
    plt.contourf(XX, YY, np.reshape(labels, (100, 100)))
    plt.scatter(X[:, 0], X[:, 1], c=y, s=3, cmap="coolwarm")
    plt.scatter(c[:, 0], c[:, 1], c="black", s=10)
    plt.show()


if __name__ == '__main__':
    cw4()
