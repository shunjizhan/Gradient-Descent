import numpy as np
from random import randint
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.preprocessing import normalize


def n(v):
    return normalize(v[:, np.newaxis], axis=0).ravel()


class GradientDescent:
    def __init__(self):
        self.n, self.d = 6000, 100
        self.X = np.random.normal(size=(self.n, self.d))
        self.beta_real = np.random.rand(self.d)
        self.Y = self.X.dot(self.beta_real)

        self.main()

    def main(self):
        beta_rand = np.random.rand(self.d)  # initial beta for GD
        errors_GD = self.GD(np.copy(beta_rand))
        errors_NAGD = self.NAGD(np.copy(beta_rand))
        errors_SGD = self.SGD(np.copy(beta_rand))

        print errors_GD[:5]
        print errors_NAGD[:5]
        print errors_SGD[:5]
        # set y axis to display integer
        # fig, ax = plt.subplots()
        # ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        plt.xlabel('iterations')
        plt.ylabel('f(x)')
        X = np.arange(50)
        plt.plot(X, errors_GD, 'r-', X, errors_NAGD, 'b-')
        plt.show()
        plt.plot(errors_SGD)
        plt.show()

    def GD(self, beta):
        step = 0.025
        errors = []
        for i in range(50):
            sqr_error_ = self.sqr_error(beta)
            errors.append(sqr_error_)
            # update beta
            beta -= step * self.gradient(beta)
        return errors

    def NAGD(self, beta):
        y, z = np.copy(beta), np.copy(beta)
        step = 0.025
        errors = []
        for i in range(50):
            sqr_error_ = self.sqr_error(beta)
            errors.append(sqr_error_)
            # update beta
            g = self.gradient(beta)
            a = 2.0 / (i + 3)
            y = beta - step * g
            z -= ((i + 1) / 2) * step * g
            beta = (1 - a) * y + a * z
        return errors

    def SGD(self, beta):
        step = 0.005
        errors = []
        for i in range(300):
            sqr_error_ = self.sqr_error(beta)
            errors.append(sqr_error_)
            # update beta
            beta = beta - step * self.stochastic_gradient(beta)
        return errors

    # gradient of f(X) = (X*Beta - y) ^ 2 / N
    def gradient(self, beta):
        X, Y = self.X, self.Y
        g = np.zeros(X.shape[1])
        for i in range(X.shape[0]):
            xi = X[i, :]
            yi = Y[i]
            g += xi * (xi.dot(beta) - yi)
        return g * 2 / X.shape[0]

    # estimator of gradient of f(X) = (X*Beta - y) ^ 2 / N
    def stochastic_gradient(self, beta):
        X, Y = self.X, self.Y
        i = randint(0, X.shape[0])
        xi_T = X[i, :]
        xi = np.transpose(xi_T)
        yi = Y[i]
        return xi * (xi_T.dot(beta) - yi) * 2

    # squared error
    def sqr_error(self, beta):
        X, Y = self.X, self.Y
        N = X.shape[0]
        temp = X.dot(beta) - Y
        return temp.dot(temp) / N


if __name__ == '__main__':
    lm = GradientDescent()
