from __future__ import print_function
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

"""
This file is part of the computer assignments for the course DD1418/DD2418 Language engineering at KTH.
Created 2017 by Johan Boye, Patrik Jonell and Dmytro Kalpakchi.
"""


class BinaryLogisticRegression(object):
    """
    This class performs binary logistic regression using batch gradient descent
    or stochastic gradient descent
    """

    #  ------------- Hyperparameters ------------------ #

    LEARNING_RATE = 0.04  # The learning rate.
    CONVERGENCE_MARGIN = 0.01  # The convergence criterion.
    # Maximal number of passes through the datapoints in stochastic gradient descent.
    MAX_EPOCHS = 10000
    # Minibatch size (only for minibatch gradient descent)
    MINIBATCH_SIZE = 1000

    # every 10 epochs we'll check if out gradients are good on the whole dataset
    CHECK_FIT_EPOCH = 10

    LAMBD = 0.0

    # ----------------------------------------------------------------------

    def __init__(self, x=None, y=None, theta=None):
        """
        Constructor. Imports the data and labels needed to build theta.

        @param x The input as a DATAPOINT*FEATURES array.
        @param y The labels as a DATAPOINT array.
        @param theta A ready-made model. (instead of x and y)
        """
        if not any([x, y, theta]) or all([x, y, theta]):
            raise Exception('You have to either give x and y or theta')

        if theta:
            self.FEATURES = len(theta)
            self.theta = theta

        elif x and y:
            # Number of datapoints.
            self.DATAPOINTS = len(x)

            # Number of features.
            self.FEATURES = len(x[0]) + 1

            # Encoding of the data points (as a DATAPOINTS x FEATURES size array).
            self.x = np.concatenate(
                (np.ones((self.DATAPOINTS, 1)), np.array(x)), axis=1)

            # Correct labels for the datapoints.
            self.y = np.array(y)

            # The weights we want to learn in the training phase.
            rng = np.random.default_rng(seed=42)
            self.theta = rng.uniform(-1, 1, self.FEATURES)

            # The current gradient.
            # NOTE: changed this to `np.ones` for convenience
            self.gradient = np.ones(self.FEATURES)

            self.rng = np.random.default_rng(seed=420)

    # ----------------------------------------------------------------------

    def sigmoid(self, z):
        """
        The logistic function.
        """
        # return np.maximum(0.0, z)
        return 1.0 / (1 + np.exp(-z))
        # if (len(z.shape) == 1):
        #     shifted = z - np.max(z)
        #     x_exp = np.exp(shifted)
        #     summ = np.sum(x_exp)
        #     return x_exp / summ
        # shifted = z - np.max(z, axis=1)

        # x_exp = np.exp(shifted)
        # summ = np.sum(x_exp, axis=1)
        # return x_exp / summ

    def conditional_prob(self, label, datapoint):
        """
        Computes the conditional probability P(label|datapoint)
        """

        # REPLACE THE COMMAND BELOW WITH YOUR CODE
        prob = self.compute_activation(self.x[datapoint])
        return prob if label == 1 else 1 - prob

    def compute_activation(self, x):
        z = x @ self.theta
        return self.sigmoid(z)

    def compute_gradients(self, x, a, y):
        return np.dot(a - y, x) / x.shape[0]

    def stochastic_fit(self):
        """
        Performs Stochastic Gradient Descent.
        """
        # self.init_plot(self.FEATURES)

        # TODO: maybe check convergence using the whole dataset like every 100th iteration

        N = self.x.shape[0]
        # YOUR CODE HERE
        epoch = 0
        # a local variable for the condition check
        gradient = np.ones(self.FEATURES)
        while np.any(np.abs(gradient) > self.CONVERGENCE_MARGIN) and epoch < self.MAX_EPOCHS:
            permutation = self.rng.permutation(N)
            self.x = self.x[permutation]
            self.y = self.y[permutation]

            for i in range(N):
                # forward prop
                a = self.compute_activation(self.x[i])
                # gradients
                self.gradient = self.compute_gradients(self.x[i], a, self.y[i])
                # backward prop
                self.theta -= self.LEARNING_RATE * self.gradient

            if epoch % self.CHECK_FIT_EPOCH == 0:
                gradient = self.compute_gradients(
                    self.x, self.compute_activation(self.x), self.y)
                print(gradient)

            print(epoch)
            epoch += 1

    def minibatch_fit(self):
        """
        Performs Mini-batch Gradient Descent.
        """
        # self.init_plot(self.FEATURES)
        # YOUR CODE HERE
        N = self.x.shape[0]

        while np.any(np.abs(self.gradient) > self.CONVERGENCE_MARGIN):

            # shuffle the dataset
            permutation = self.rng.permutation(N)
            self.x = self.x[permutation]
            self.y = self.y[permutation]

            # iterate through batches
            for i in range(0, N, self.MINIBATCH_SIZE):
                j = min(i + self.MINIBATCH_SIZE, N)
                # batch size
                batch_x = self.x[i:j]
                batch_y = self.y[i:j]

                # forward prop
                a = self.compute_activation(batch_x)
                # gradients
                self.gradient = self.compute_gradients(
                    batch_x, a, batch_y) + 2 * self.LAMBD * self.gradient
                # backward prop
                self.theta -= self.LEARNING_RATE * self.gradient
            print(self.gradient)

    def fit(self):
        """
        Performs Batch Gradient Descent
        """
        # self.init_plot(self.FEATURES)

        # YOUR CODE HERE
        while np.any(np.abs(self.gradient) > self.CONVERGENCE_MARGIN):
            # forward prop
            a = self.compute_activation(self.x)
            # gradients
            self.gradient = self.compute_gradients(self.x, a, self.y)
            # backward prop
            self.theta -= self.LEARNING_RATE * self.gradient

    def classify_datapoints(self, test_data, test_labels):
        """
        Classifies datapoints
        """
        print('Model parameters:')

        print('  '.join('{:d}: {:.4f}'.format(
            k, self.theta[k]) for k in range(self.FEATURES)))

        self.DATAPOINTS = len(test_data)

        self.x = np.concatenate(
            (np.ones((self.DATAPOINTS, 1)), np.array(test_data)), axis=1)
        self.y = np.array(test_labels)
        confusion = np.zeros((self.FEATURES, self.FEATURES))

        for d in range(self.DATAPOINTS):
            prob = self.conditional_prob(1, d)
            predicted = 1 if prob > .5 else 0
            confusion[predicted][self.y[d]] += 1

        print('                       Real class')
        print('                 ', end='')
        print(' '.join('{:>8d}'.format(i) for i in range(2)))
        for i in range(2):
            if i == 0:
                print('Predicted class: {:2d} '.format(i), end='')
            else:
                print('                 {:2d} '.format(i), end='')
            print(' '.join('{:>8.3f}'.format(
                confusion[i][j]) for j in range(2)))

        acc = (confusion[0][0] + confusion[1][1]) / self.DATAPOINTS
        # precision = TP / (TP + FP)
        precision = confusion[1][1] / (confusion[1][1] + confusion[1][0])

        print(f'Accuracy: {acc}, Precision: {precision}')

    def print_result(self):
        print(' '.join(['{:.2f}'.format(x) for x in self.theta]))
        print(' '.join(['{:.2f}'.format(x) for x in self.gradient]))

    # ----------------------------------------------------------------------

    def update_plot(self, *args):
        """
        Handles the plotting
        """
        if self.i == []:
            self.i = [0]
        else:
            self.i.append(self.i[-1] + 1)

        for index, val in enumerate(args):
            self.val[index].append(val)
            self.lines[index].set_xdata(self.i)
            self.lines[index].set_ydata(self.val[index])

        self.axes.set_xlim(0, max(self.i) * 1.5)
        self.axes.set_ylim(0, max(max(self.val)) * 1.5)

        plt.draw()
        plt.pause(1e-20)

    def init_plot(self, num_axes):
        """
        num_axes is the number of variables that should be plotted.
        """
        self.i = []
        self.val = []
        plt.ion()
        self.axes = plt.gca()
        self.lines = []

        for i in range(num_axes):
            self.val.append([])
            self.lines.append([])
            self.lines[i], = self.axes.plot(
                [], self.val[0], '-', c=[random.random() for _ in range(3)], linewidth=1.5, markersize=4)

    # ----------------------------------------------------------------------


def main():
    """
    Tests the code on a toy example.
    """
    x = [
        [1, 1], [0, 0], [1, 0], [0, 0], [0, 0], [0, 0],
        [0, 0], [0, 0], [1, 1], [0, 0], [0, 0], [1, 0],
        [1, 0], [0, 0], [1, 1], [0, 0], [1, 0], [0, 0]
    ]

    #  Encoding of the correct classes for the training material
    y = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0]
    b = BinaryLogisticRegression(x, y)
    b.fit()
    b.print_result()


if __name__ == '__main__':
    main()
