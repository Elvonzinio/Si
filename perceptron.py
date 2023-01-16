from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


class SimplePerceptron(BaseEstimator, ClassifierMixin):
    classLabels = None
    weights = None

    def __init__(self, learningRate=1.0, m=20):
        self.learningRate = learningRate
        self.m = m

    def fit(self, X, y, maxIteration=500):
        self.classLabels = np.unique(y)
        samplesNumber, featuresNumber = X.shape

        yy = np.ones(samplesNumber, dtype="int8")
        yy[y == self.classLabels[0]] = -1

        X = np.c_[np.ones(samplesNumber), X]
        self.weights = np.zeros(featuresNumber + 1)
        iteration = 0
        while iteration < maxIteration:
            error_indexes = []
            for i, x in enumerate(X):
                s = self.weights.dot(x)

                if s > 0.0:
                    f = 1
                else:
                    f = -1

                if f != yy[i]:
                    error_indexes.append(i)
                    break
            if len(error_indexes) == 0:
                break
            i = error_indexes[np.random.randint(len(error_indexes))]
            self.weights = self.weights + self.learningRate * yy[i] * X[i]
            iteration += 1

    def predict(self, X):
        s = self.decision_function(X)

        predictions = []
        for element in s:
            if element > 0.0:
                predictions.append(1)
            else:
                predictions.append(0)
        predictions = np.array(predictions)

        return self.classLabels[predictions]

    def decision_function(self, X):
        m = X.shape[0]
        X = np.c_[np.ones(m), X]
        return self.weights.dot(X.T)  # zwraca sume iloczynow

    def distance(self, x, c, sig=0.2):
        z = np.zeros((len(x), self.m))
        for i, elem_x in enumerate(x):
            for j in range(self.m):
                for element, value_c in zip(elem_x, c[j]):
                    z[i, j] += (element - value_c) ** 2
                z[i, j] = np.exp(-z[i, j] / (2 * sig ** 2))
        return z

    def centers(self):
        X = np.random.uniform(-1, 1, self.m)
        Y = np.random.uniform(-1, 1, self.m)
        c = np.c_[X, Y]
        return c

    def contour(self, c, size=100):
        XY = np.linspace(-1, 1, size)
        XX, YY = np.meshgrid(XY, XY)

        XXres = np.reshape(XX, size ** 2)
        YYres = np.reshape(YY, size ** 2)

        XY = np.c_[XXres, YYres]

        z = self.distance(XY, c)
        labels = self.predict(z)

        return labels, XX, YY
