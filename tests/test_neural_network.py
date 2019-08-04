from src.neural_network import NeuralNetwork
from src import activation_functions


class passthrough():
    def f(x):
        return x

    def df(x):
        return 1


def test_str():
    print(NeuralNetwork([3,3,3,1]))

def test_load_1():
    n = NeuralNetwork(
        [3,2,1],
        inner_activation_function = passthrough,
        outer_activation_function = passthrough,
        initializer=lambda: 1,
        num_iterations=10
        )
    n.weights[0][0][1] = 10
    n.weights[0][1][1] = 10
    n.weights[0][2][1] = 10
    n.weights[0][3][1] = 10

    print('')
    X = [[1,1,0.5,]]
    assert [39.5] == n.load(X[0])

    n.train(X, [3.5])
    n.print_debug()

def test_load_gaussian(reset_random_seed):
    n = NeuralNetwork(
        [3,2,3,1],
        num_iterations=1000000
    )
    print(n.weights[0][0][1])
    X = [
        [1,1,1,],
        [2,2,2,],
    ]
    y = [0.5, 0.2]
    assert [0.3286929178446704] == n.load(X[0])

def test_train_gaussian(reset_random_seed):
    n = NeuralNetwork(
        [3,2,3,1],
        num_iterations=1000
    )
    print(n.weights[0][0][1])
    X = [
        [1,1,1,],
        [2,2,2,],
    ]
    y = [0.5, 0.2]
    n.train(X, y)
    n.print_debug()
