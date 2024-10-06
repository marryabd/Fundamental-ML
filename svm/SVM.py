import numpy as np


class SVM:
    def __init__(self, C=100, lr_rate=0.001, n_iter=1000):
        self.C = C
        self.lr_rate = lr_rate
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_sample, n_feature = X.shape
        # initialize
        self.weights = np.random.randn(n_feature) * 0.01
        self.bias = 0
        y_ = np.where(y<=0, -1, 1)

        for _ in range(self.n_iter):

            condition = y_ * (np.dot(X, self.weights) + self.bias) < 1

            self.weights -= self.lr_rate * (2 * self.weights - self.C* np.dot(X.T, y_*condition.astype(int)))
            self.bias += np.sum(self.lr_rate*self.C*y_*condition.astype(int))

    def predict(self, X):
        return np.where(np.dot(X, self.weights) + self.bias <=0, -1, 1)