from .matrix import Matrix
from . import activation_functions


class NeuralNetwork(object):

    def __init__(self, layer_widths,
        activation_function=activation_functions.reLU):
        self.matrices = []
        for i in range(len(layer_widths) -1):
            self.matrices.append(
                Matrix.create([layer_widths[i], layer_widths[i+1]], 1)
            )
        self.activation_function = activation_function

    def __str__(self):
        return " > ".join([",".join([str(x) for x in m.dimensions]) for m in self.matrices])

    def load(self, input_list):
        # Must be the same length as the first layer's width!
        result = Matrix([input_list])
        for m in self.matrices:
            result = result.multiply(m)
            result.apply_function(self.activation_function)

        return result._arr[0]
