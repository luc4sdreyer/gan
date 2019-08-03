"""
For some background on backpropagation, see:
https://web.archive.org/web/20181116193650/https://brilliant.org/wiki/backpropagation/

The weight matrix convention used here is:

Weight from node i to node j at layer k: self.matrices[k][i][j]

with the first node being the bias, except for the output layer that doesn't have a bias.
"""

import random

from .matrix import Matrix
from . import activation_functions


class NeuralNetwork(object):

    def __init__(self, layer_widths,
        activation_function=activation_functions.logistic,
        initializer=lambda: random.gauss(0, 1)):

        self.depth = len(layer_widths)
        self.activation_function = activation_function

        # +1 for bias nodes, but not on the final layer
        self.layer_widths = []
        for i in range(self.depth):
            width = layer_widths[i]
            width = width + 1 if i < len(layer_widths) - 1 else width
            self.layer_widths.append(width)

        self.matrices = []
        for i in range(self.depth -1):
            m = Matrix.create([self.layer_widths[i], self.layer_widths[i+1]], initializer)
            self.matrices.append(m)

        # bias nodes' inputs are always zero
        for i in range(self.depth -2):
            m = self.matrices[i]
            for j in range(len(m._arr)):
                m[j][0] = 0

        # backpropagation info
        self._a = [None] * self.depth  # node inputs
        self._o = [None] * self.depth  # node outputs


    def __str__(self):
        return " > ".join([",".join([str(x) for x in m.dimensions]) for m in self.matrices])

    def print_debug(self):
        for m in self.matrices:
            m.print_debug()

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
        for i in range(self.depth -1):
            m = self.matrices[i]
            input_list = input_list.multiply(m)
            self._a[i] = input_list._arr.copy()
            input_list.apply_function(self.activation_function)
            self._o[i] = input_list._arr.copy()
            # set bias output
            if i < len(self.matrices) -1:
                input_list._arr[0][0] = 1.0

            print(input_list)

        return input_list._arr[0]
