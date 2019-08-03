from src.neural_network import NeuralNetwork
from src import activation_functions


def test_str():
    print(NeuralNetwork([3,3,3,1]))

def test_load():
    n = NeuralNetwork([3,3,3,2], activation_function = activation_functions.reLU)
    assert [27, 27] == n.load([1,1,1,])
