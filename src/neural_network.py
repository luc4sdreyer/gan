"""
For some background on backpropagation, see:
https://web.archive.org/web/20181116193650/https://brilliant.org/wiki/backpropagation/

The weight matrix convention used here is:

Weight from node i to node j at layer k: self.weights[k][i][j]

with the first node being the bias, except for the output layer that doesn't have a bias.

This network uses mean squared error for the error (loss) function.
"""

import random

from .matrix import Matrix
from . import activation_functions


class NeuralNetwork(object):

    def __init__(self, layer_widths,
        inner_activation_function=activation_functions.logistic,
        outer_activation_function=activation_functions.logistic,
        initializer=lambda: random.gauss(0, 1),
        num_iterations=1000,
        learning_rate=0.1):

        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.depth = len(layer_widths)
        self.inner_activation_function = inner_activation_function
        self.outer_activation_function = outer_activation_function

        # +1 for bias nodes, but not on the final layer
        self.layer_widths = []
        for i in range(self.depth):
            width = layer_widths[i]
            width = width + 1 if i < len(layer_widths) - 1 else width
            self.layer_widths.append(width)

        self.weights = []
        for i in range(self.depth -1):
            w = Matrix.create([self.layer_widths[i], self.layer_widths[i+1]], initializer)
            self.weights.append(w)

        # bias nodes' inputs are always zero
        for i in range(self.depth -2):
            w = self.weights[i]
            for j in range(len(w._arr)):
                w[j][0] = 0

        # backpropagation info
        self._a = [None] * self.depth  # node inputs
        self._o = [None] * self.depth  # node outputs
        self.deltas = []


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

    def _get_delta(self, X, y):
        # get the error (delta), working backwards from the output layer
        delta = [None] * self.depth
        for current_depth in range(self.depth -1, 0, -1):
            # print("current_depth: %s" % current_depth)
            current_width = self.layer_widths[current_depth]
            # print("current_width: %s" % current_width)
            if current_depth == self.depth -1:
                # print("self._a: %s" % self._a)
                delta_layer = Matrix([[self.outer_activation_function.df(self._a[current_depth][0])]])
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
        Partial derivative of the error with respect to each weight (pdew).
        """
        # print("pdews")
        # print(self._o)
        # print(deltas)
        pdews = [None] * self.depth
        for i in range(0, self.depth -1):
            pdews[i] = Matrix([self._o[i]]).transpose().multiply(deltas[i+1].transpose())
        return pdews

    def train_step(self, X, y):
        self.total_pdews = None
        for i in range(len(X)):
            actual_y = self.load(X[i])
            self.deltas = self._get_delta(X, y)
            self.pdews = self._get_pdew(self.deltas)
            if i == 0:
                self.total_pdews = self.pdews
            else:
                for j in range(self.dept -1):
                    self.total_pdews[j].add(self.pdews[j])

        for j in range(0, self.depth -1):
            self.total_pdews[j].multiply_scalar(1.0 / len(X) * -self.learning_rate)

        self.weight_adjustments = self.total_pdews

        for j in range(0, self.depth -1):
            self.weights[j].add(self.weight_adjustments[j])


    def train(self, X, y):
        for epoch in range(self.num_iterations):
            self.train_step(X, y)
