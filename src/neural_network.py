"""
For some background on backpropagation, see:
https://brilliant.org/wiki/backpropagation/
http://neuralnetworksanddeeplearning.com/chap3.html

The weight matrix convention used here is:

Weight from node i to node j at layer k: self.weights[k][i][j]

with the first node being the bias, except for the output layer that doesn't have a bias.

This network uses mean squared error for the error (loss) function.
"""

import sys
import random
import time

from .matrix import Matrix
from . import activation_functions


class NeuralNetwork(object):

    def __init__(self, layer_widths,
        inner_activation_function=activation_functions.logistic,
        outer_activation_function=activation_functions.logistic,
        initializer=lambda: random.gauss(0, 1),
        num_iterations=1000,
        learning_rate=0.1,
        stop_criterion=lambda err: err < 10**-5,
        restart_limit=0,
        lamb=1.0,):

        self.lamb = lamb
        self.restart_limit = restart_limit
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations if num_iterations > 0 else sys.maxint
        self.depth = len(layer_widths)
        self.inner_activation_function = inner_activation_function
        self.outer_activation_function = outer_activation_function
        self.stop_criterion = stop_criterion
        self.initializer = initializer

        # +1 for bias nodes, but not on the final layer
        self.layer_widths = []
        for i in range(self.depth):
            width = layer_widths[i]
            width = width + 1 if i < len(layer_widths) - 1 else width
            self.layer_widths.append(width)

        self.restart()

    def restart(self):
        self.weights = []
        for i in range(self.depth -1):
            w = Matrix.create([self.layer_widths[i], self.layer_widths[i+1]], self.initializer)
            self.weights.append(w)

        self.zero_bias_inputs()

        # backpropagation info
        self._a = [None] * self.depth  # node inputs
        self._o = [None] * self.depth  # node outputs
        self.deltas = []

        self.min_error = sys.float_info.max

        self.last_output_time = None

    def zero_bias_inputs(self):
        # bias nodes' inputs are always zero
        for i in range(self.depth -2):
            w = self.weights[i]
            for j in range(len(w._arr)):
                w[j][0] = 0

    def __str__(self):
        return " > ".join([",".join([str(x) for x in m.dimensions]) for m in self.weights])

    def print_debug(self):
        print("Weights:")
        for w in self.weights:
            w.print_debug()

        if self.deltas:
            print("Deltas:")
            for d in self.deltas:
                if d:
                    d.print_debug()

    def load(self, input_list):
        """
        input_list must be the same length as the first layer's width!
        Input is considered a 1 x len(input_list) matrix. In mathematical matrix notation:
        [i0 i1 i2 ... ]
        +1 to the input for the bias node
        """
        result = Matrix([[1] + input_list])
        return self.feedforward(result)

    def feedforward(self, input_list):
        self._o[0] = input_list._arr.copy()[0]
        for i in range(self.depth -1):
            w = self.weights[i]
            input_list = input_list.multiply(w)
            self._a[i+1] = input_list._arr.copy()[0]

            # Are we in the final (output) layer?
            if i < self.depth -2:
                input_list.apply_function(self.inner_activation_function.f)
                # Pin the bias node's output
                input_list[0][0] = 1.0
            else:
                input_list.apply_function(self.outer_activation_function.f)

            self._o[i+1] = input_list._arr.copy()[0]

            # print(input_list)

        return input_list._arr[0]

    def _get_delta(self, y_diff):
        # get the error (delta), working backwards from the output layer
        delta = [None] * self.depth
        for current_depth in range(self.depth -1, 0, -1):
            # print("current_depth: %s" % current_depth)
            current_width = self.layer_widths[current_depth]
            # print("current_width: %s" % current_width)
            if current_depth == self.depth -1:
                # print("self._a: %s" % self._a)
                delta_layer = Matrix([[
                    self.outer_activation_function.df(self._a[current_depth][0]) * y_diff
                ]])
            else:
                activation_matrix = Matrix.create([current_width, current_width], lambda: 0.0)
                for i in range(current_width):
                    # print("self._a[current_depth]: %s" % self._a[current_depth])
                    activation_matrix[i][i] = self.inner_activation_function.df(self._a[current_depth][i])

                delta_layer = activation_matrix.multiply(self.weights[current_depth]).multiply(delta[current_depth + 1])

            delta[current_depth] = delta_layer
        return delta

    def _get_pdew(self, deltas):
        """
        Partial derivative of the error with respect to each weight (pdew), or ∂E/∂w
        """
        # print("pdews")
        # print(self._o)
        # print(deltas)
        pdews = [None] * self.depth
        for i in range(0, self.depth -1):
            pdews[i] = Matrix([self._o[i]]).transpose().multiply(deltas[i+1].transpose())
        return pdews

    def train_step(self, X, y, epoch):
        self.total_pdews = None
        mean_squared_error = 0

        # regularization
        reg = 0
        for j in range(0, self.depth -1):
            reg += self.weights[j].squared_l2_norm()
        reg = self.lamb / (2.0 * len(X))

        # get partial errors
        for i in range(len(X)):
            actual_y = self.load(X[i])
            y_diff = actual_y[0] - y[i]
            mean_squared_error += y_diff * y_diff
            total_cost = y_diff + reg
            self.deltas = self._get_delta(total_cost)
            self.pdews = self._get_pdew(self.deltas)
            if i == 0:
                self.total_pdews = self.pdews
            else:
                for j in range(self.depth -1):
                    self.total_pdews[j].add(self.pdews[j])

        # apply learning rate
        for j in range(0, self.depth -1):
            self.total_pdews[j].multiply_scalar(-self.learning_rate / len(X))

        self.weight_adjustments = self.total_pdews

        # adjust weights
        for j in range(0, self.depth -1):
            # regularization
            self.weights[j].multiply_scalar(1.0 - self.lamb * self.learning_rate / len(X))
            # rest of the error
            self.weights[j].add(self.weight_adjustments[j])

        self.zero_bias_inputs()

        mean_squared_error /= len(X)

        if mean_squared_error < self.min_error:
            self.min_error = mean_squared_error
            self.best_epoch = epoch

        if not self.last_output_time or self.last_output_time + 3 < time.time():
            self.last_output_time = time.time()
            print("Epoch %s \tMSE: %.5f \t best MSE: %.5f" % (epoch, mean_squared_error, self.min_error))

        # self.print_debug()

        if self.stop_criterion(mean_squared_error):
            print("Stopping at epoch %s due to stop stop criterion" % epoch)
            return False

        if self.restart_limit > 0 and epoch - self.best_epoch > self.restart_limit:
            print("Restarting at epoch %s due to no improvement in %s epochs" % (epoch, self.restart_limit))
            self.restart()

        return True

    def train(self, X, y):
        for epoch in range(self.num_iterations):
            if not self.train_step(X, y, epoch):
                break

    def test(self, X, y):
        mean_squared_error = 0

        for i in range(len(X)):
            actual_y = self.load(X[i])
            y_diff = actual_y[0] - y[i]
            mean_squared_error += y_diff * y_diff

        mean_squared_error /= len(X)

        print("test MSE: %.5f" % (mean_squared_error))
