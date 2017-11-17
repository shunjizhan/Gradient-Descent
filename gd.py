import numpy as np
from random import randint
import matplotlib.pyplot as plt


# gradient of f(X) = (X*Beta - y) ^ 2 / N
def gradient(X, Y, beta):
    g = np.zeros(X.shape[1])
    for i in range(X.shape[0]):
        xi_T = X[i, :]
        xi = np.transpose(xi_T)
        yi = Y[i]
        g += xi * (xi_T.dot(beta) - yi) * 2 / X.shape[0]
    return g


# estimator of gradient of f(X) = (X*Beta - y) ^ 2 / N
def stochastic_gradient(X, Y, beta):
    i = randint(0, X.shape[0])
    xi_T = X[i, :]
    xi = np.transpose(xi_T)
    yi = Y[i]
    return xi * (xi_T.dot(beta) - yi) * 2 / X.shape[0]


def sqr_error(X, Y, beta):
    temp = X.dot(beta) - Y
    return temp.dot(np.transpose(temp)) / X.shape[0]


# Linear Regression implementation
class LinearRegression:
    def __init__(self):
        self.step_size = 0.025
        self.n, self.d = 6000, 100

        self.X = np.random.rand(self.n, self.d)
        self.beta_real = np.random.rand(self.d)
        self.Y = self.X.dot(self.beta_real)

        self.beta_rand = np.random.rand(self.d)   # initial beta for GD
        self.GD()
        self.SGD()

    def GD(self):
        beta_GD = self.beta_rand
        errors_GD = []
        for i in range(50):
            sqr_error_GD = sqr_error(self.X, self.Y, beta_GD)
            errors_GD.append(sqr_error_GD)
            beta_GD = beta_GD - self.step_size * gradient(self.X, self.Y, beta_GD)
        print errors_GD
        plt.plot(errors_GD, 'r')
        plt.show()

    def SGD(self):
        beta_SGD = self.beta_rand
        errors_SGD = []
        for i in range(250):
            sqr_error_SGD = sqr_error(self.X, self.Y, beta_SGD)
            errors_SGD.append(sqr_error_SGD)
            beta_SGD = beta_SGD - self.step_size * stochastic_gradient(self.X, self.Y, beta_SGD)
        plt.plot(errors_SGD, 'b')
        plt.show()


if __name__ == '__main__':
    lm = LinearRegression()