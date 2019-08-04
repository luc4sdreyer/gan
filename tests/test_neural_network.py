from src.neural_network import NeuralNetwork
from src import activation_functions

from sklearn import datasets
from sklearn.model_selection import train_test_split


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
        num_iterations=1000,
        lamb = 0.0,
    )
    print(n.weights[0][0][1])
    X = [
        [1,1,1,],
        [2,2,2,],
    ]
    y = [0.5, 0.2]
    n.train(X, y)
    n.print_debug()
    assert n.min_error == 0.023915334328727347

def test_train_iris(reset_random_seed):
    iris = datasets.load_iris()
    orig_X = iris.data.tolist()
    orig_y = iris.target.tolist()

    # discard the 3rd class
    X = []
    y = []
    for i in range(len(orig_y)):
        if orig_y[i] <= 1:
            X.append(orig_X[i])
            y.append(orig_y[i])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33)

    n = NeuralNetwork(
        [len(X_train[0]),5,5,1],
        num_iterations=1,
        inner_activation_function = activation_functions.reLU,
        outer_activation_function = activation_functions.logistic,
        stop_criterion=lambda err: err < 10**-4,
    )
    n.train(X_train, y_train)
    n.test(X_test, y_test)
    n.print_debug()
