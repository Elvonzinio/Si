from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


class SimplePerceptron(BaseEstimator, ClassifierMixin):
    classLabels = None
    weights = None

    def __init__(self, learningRate=1.0, m=20):
        self.learningRate = learningRate
        self.m = m

    def fit(self, X, y, maxIterations=500):
        self.classLabels = np.unique(y)
        samplesNumber, featuresNumber = X.shape

        X = np.c_[np.ones(samplesNumber), X]
        self.weights = np.zeros(featuresNumber + 1)  # czemu tu +1
        iterations = 0
        while iterations < maxIterations:
            errorIndexes = []  # lista indeksow ktore nie sa prawidlowo predictowane
            for errorIndex, x in enumerate(X):
                prediction = self.weights.dot(x)

                if prediction > 0.0:
                    predResult = 1
                else:
                    predResult = -1

                if predResult != y[errorIndex]:  # czy predykcja jest prawidlowa
                    errorIndexes.append(errorIndex)
                    break
            if len(errorIndexes) == 0:
                break
            errorIndex = errorIndexes[np.random.randint(len(errorIndexes))]
            self.weights = self.weights + self.learningRate * y[errorIndex] * X[errorIndex]
            iterations += 1

        return self.weights, iterations

    def predict(self, X):
        prediction = self.decisionFunction(X)

        predictions = np.zeros(len(X), dtype="int8")
        for index, element in enumerate(prediction):  # Przypisanie etykiet do wyniku predykcji
            if element > 0.0:
                predictions[index] = 1

        return self.classLabels[predictions]

    def decisionFunction(self, X):
        samplesNumber = X.shape[0]
        X = np.c_[np.ones(samplesNumber), X]  # laczenie tablic
        return self.weights.dot(X.T)  # zwraca sume iloczynow

    def distance(self, matrix, centroidsMatrix, sig=0.2):
        distances = np.zeros((len(matrix), self.m))
        for i, sample in enumerate(matrix):
            for j in range(self.m):  # czym jest self.m ?
                for z, c in zip(sample, centroidsMatrix[j]):
                    distances[i, j] += (z - c) ** 2
                distances[i, j] = np.exp(-distances[i, j] / (2 * sig ** 2))  # link do wzoru
        return distances

    def centroids(self):
        X = np.random.uniform(-1, 1, self.m)
        Y = np.random.uniform(-1, 1, self.m)
        centroidMatrix = np.c_[X, Y]  # konkatonacja tablic (link do exampla)
        return centroidMatrix

    def contour(self, centroids, size=100):
        XY = np.linspace(-1, 1, size)
        XX, YY = np.meshgrid(XY, XY)

        XXres = np.reshape(XX, size ** 2)
        YYres = np.reshape(YY, size ** 2)

        XY = np.c_[XXres, YYres]

        distances = self.distance(XY, centroids)
        labels = self.predict(distances)

        return labels, XX, YY
