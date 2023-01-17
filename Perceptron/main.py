import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from perceptron import SimplePerceptron


def generateData4(size):
    X, y = make_blobs(
        n_samples=size,
        centers=2,
        n_features=2,
        cluster_std=0.5,
        random_state=0
    )

    theta = 1
    rotateMatrix = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    X = np.matmul(X, rotateMatrix)  # mnożenie macierzy

    for i, element in enumerate(y):
        if element == 0:
            y[i] = -1

    return X, y


def generateData5(size):
    np.random.seed(0)
    coordinateX = np.random.uniform(0, 2 * np.pi, size)  # uniform -> rozkład normalny
    coordinateY = np.random.uniform(-1, 1, size)

    X = np.c_[coordinateX, coordinateY]
    y = np.ones(size)

    for i, sample in enumerate(X):
        if np.abs(np.sin(sample[0])) > np.abs(sample[1]):  # link do wzoru
            y[i] = -1

    for i in range(size):
        X[i, 0] = (X[i, 0] / np.pi) - 1

    return X, y


def cw4():
    print("Zadanie1")
    X, y = generateData4(1000)
    clf = SimplePerceptron(learningRate=1.0)
    weights, iteration = clf.fit(X, y)
    print(f"Weights: {weights}")
    print(f"Iterations: {iteration}")

    triangles = []
    circles = []

    for sample, label in zip(X, y):
        if label == 1:
            triangles.append(sample)
        else:
            circles.append(sample)

    triangles = np.array(triangles)
    circles = np.array(circles)

    plt.scatter(triangles[:, 0], triangles[:, 1], c='red', s=10, marker="^")
    plt.scatter(circles[:, 0], circles[:, 1], c='blue', s=10, marker="o")

    minValue = min(X[:, 0])
    maxValue = max(X[:, 0])

    x1 = np.array([minValue, maxValue])
    x2 = (-(weights[0] + weights[1] * x1) / weights[2])
    plt.plot(x1, x2)
    plt.show()


def cw5():
    print("\n\nZadanie2")
    X, y = generateData5(1000)
    clf = SimplePerceptron(learningRate=1.0, m=100)
    centroids = clf.centroids()
    distances = clf.distance(X, centroids)
    weights, iterations = clf.fit(distances, y, maxIterations=4000)

    print(f"Iterations: {iterations}")
    print(clf.score(distances, y))

    labels, XX, YY = clf.contour(centroids, 100)
    plt.contourf(XX, YY, np.reshape(labels, (100, 100)))
    plt.scatter(X[:, 0], X[:, 1], c=y, s=3, cmap="bwr")
    plt.scatter(centroids[:, 0], centroids[:, 1], c="black", s=10)
    plt.show()


if __name__ == '__main__':
    cw4()
    cw5()
