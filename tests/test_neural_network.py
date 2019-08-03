from src.neural_network import NeuralNetwork
from src import activation_functions


def passthrough(x):
    return x


def passthrough_derivative(x):
    return 1


def test_str():
    print(NeuralNetwork([3,3,3,1]))

def test_load_1():
    n = NeuralNetwork(
        [3,2,1],
        activation_function = passthrough,
        initializer=lambda: 1
        )
    n.matrices[0][0][1] = 10
    n.matrices[0][1][1] = 10
    n.matrices[0][2][1] = 10
    n.matrices[0][3][1] = 10

    print('')
    n.print_debug()
    assert [39.5] == n.load([1,1,0.5,])

def test_load_gaussian(reset_random_seed):
    n = NeuralNetwork(
        [3,2,3,1],
        )
    print(n.matrices[0][0][1])
    n.print_debug()
    assert [0.3286929178446704] == n.load([1,1,1,])
