import random
import numpy as np


class LogisticRegression(object):
    """
    This class performs logistic regression using batch gradient descent
    or stochastic gradient descent
    """

    def __init__(self, theta=None):
        """
        Constructor. Imports the data and labels needed to build theta.
        @param theta    A ready-made model
        """
        theta_check = theta is not None

        if theta_check:
            self.FEATURES = len(theta)
            self.theta = theta

        #  ------------- Hyperparameters ------------------ #
        self.LEARNING_RATE = 0.1            # The learning rate.
        self.MINIBATCH_SIZE = 256           # Minibatch size
        # A max number of consequent epochs with monotonously
        self.PATIENCE = 5
        self.MIN_DELTA = 1e-6
        # increasing validation loss for declaring overfitting
        # ----------------------------------------------------------------------

    def init_params(self, x, y):
        """
        Initializes the trainable parameters of the model and dataset-specific variables
        """
        # To limit the effects of randomness
        np.random.seed(524287)

        # Number of features
        self.FEATURES = len(x[0]) + 1

        # Number of classes
        self.CLASSES = len(np.unique(y))

        # Training data is stored in self.x (with a bias term) and self.y
        self.x, self.y, self.xv, self.yv = self.train_validation_split(
            np.concatenate((np.ones((len(x), 1)), x), axis=1), y)

        # Number of datapoints.
        self.TRAINING_DATAPOINTS = len(self.x)

        # The weights we want to learn in the training phase.
        K = np.sqrt(1 / self.FEATURES)
        self.theta = np.random.uniform(-K, K, (self.FEATURES, self.CLASSES))

        # The current gradient.
        self.gradient = np.zeros((self.FEATURES, self.CLASSES))

        print("NUMBER OF DATAPOINTS: {}".format(self.TRAINING_DATAPOINTS))
        print("NUMBER OF CLASSES: {}".format(self.CLASSES))

    def train_validation_split(self, x, y, ratio=0.9):
        """
        Splits the data into training and validation set, taking the `ratio` * 100 percent of the data for training
        and `1 - ratio` * 100 percent of the data for validation.
        @param x        A (N, D + 1) matrix containing training datapoints
        @param y        An array of length N containing labels for the datapoints
        @param ratio    Specifies how much of the given data should be used for training
        """
        indices = np.arange(len(x))
        np.random.shuffle(indices)
        split_idx = int(len(x) * ratio)
        train_indices, val_indices = indices[:split_idx], indices[split_idx:]

        x_train, x_val = x[train_indices], x[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]

        return x_train, y_train, x_val, y_val

    def softmax(self, x):
        logits = np.dot(x, self.theta)
        logits_exp = np.exp(logits)
        probs = logits_exp / np.sum(logits_exp, axis=1, keepdims=True)

        return probs

    def loss(self, x, y):
        """
        Calculates the loss for the datapoints present in `x` given the labels `y`.
        """
        probs = self.softmax(x)
        log_probs = np.log(probs[range(len(y)), y])
        return -np.mean(log_probs)

    def conditional_log_prob(self, label, datapoint):
        """
        Computes the conditional log-probability log[P(label|datapoint)]
        """
        logits = np.dot(datapoint, self.theta)
        logits_exp = np.exp(logits)
        probs = logits_exp / np.sum(logits_exp)
        return np.log(probs[label])

    def compute_gradient(self, minibatch):
        """
        Computes the gradient based on a mini-batch
        """
        x, y = minibatch
        probs = self.softmax(x)
        probs[range(len(y)), y] -= 1
        gradient = np.dot(x.T, probs) / len(y)
        return gradient

    def fit(self, x, y):
        """
        Performs Mini-batch Gradient Descent.

        :param      x:      Training dataset (features)
        :param      y:      The list of training labels
        """
        self.init_params(x, y)

        best_val_loss = float('inf')
        patience_counter = 0
        epoch = 0

        while patience_counter < self.PATIENCE:
            epoch += 1
            minibatches = self.create_minibatches(
                self.x, self.y, self.MINIBATCH_SIZE)

            for minibatch in minibatches:
                self.gradient = self.compute_gradient(minibatch)
                self.theta -= self.LEARNING_RATE * self.gradient

            val_loss = self.loss(self.xv, self.yv)

            print("Epoch: {}, Validation Loss: {:.4f}".format(
                epoch, val_loss))

            if val_loss < best_val_loss and best_val_loss - val_loss > self.MIN_DELTA:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

    def create_minibatches(self, x, y, batch_size):
        indices = np.random.permutation(len(x))
        minibatches = [(x[indices[i:i + batch_size]], y[indices[i:i + batch_size]])
                       for i in range(0, len(x), batch_size)]
        return minibatches

    def get_log_probs(self, x):
        """
        Get the log-probabilities for all labels for the datapoint `x`

        :param      x:    a datapoint
        """
        if self.FEATURES - len(x) == 1:
            x = np.array(np.concatenate(([1.], x)))
        else:
            raise ValueError("Wrong number of features provided!")
        return [self.conditional_log_prob(c, x) for c in range(self.CLASSES)]

    def classify_datapoints(self, x, y):
        """
        Classifies datapoints
        """
        confusion = np.zeros((self.CLASSES, self.CLASSES))

        x = np.concatenate((np.ones((len(x), 1)), x), axis=1)

        no_of_dp = len(y)
        for d in range(no_of_dp):
            best_prob, best_class = -float('inf'), None
            for c in range(self.CLASSES):
                prob = self.conditional_log_prob(c, x[d])
                if prob > best_prob:
                    best_prob = prob
                    best_class = c
            confusion[best_class][y[d]] += 1

        print('                       Real class')
        print('                 ', end='')
        print(' '.join('{:>8d}'.format(i) for i in range(self.CLASSES)))
        for i in range(self.CLASSES):
            if i == 0:
                print('Predicted class: {:2d} '.format(i), end='')
            else:
                print('                 {:2d} '.format(i), end='')
            print(' '.join('{:>8.3f}'.format(
                confusion[i][j]) for j in range(self.CLASSES)))
            acc = sum([confusion[i][i]
                      for i in range(self.CLASSES)]) / no_of_dp
            print("Accuracy: {0:.2f}%".format(acc * 100))

    def print_result(self):
        print(' '.join(['{:.2f}'.format(x) for x in self.theta]))
        print(' '.join(['{:.2f}'.format(x) for x in self.gradient]))


def main():
    """
    Tests the code on a toy example.
    """
    def get_label(dp):
        if dp[0] == 1:
            return 2
        elif dp[2] == 1:
            return 1
        else:
            return 0

    from itertools import product
    x = np.array(list(product([0, 1], repeat=6)))

    #  Encoding of the correct classes for the training material
    y = np.array([get_label(dp) for dp in x])

    ind = np.arange(len(y))

    np.random.seed(524287)
    np.random.shuffle(ind)

    b = LogisticRegression()
    b.fit(x[ind][:-20], y[ind][:-20])
    b.classify_datapoints(x[ind][-20:], y[ind][-20:])


if __name__ == '__main__':
    main()
