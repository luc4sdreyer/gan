from math import *


def tanH(x):
    return tanh(x)


def tanH_derivative(x):
    return 1.0 - (tanH(x) * tanH(x))


def logistic(x):
    """
    (a.k.a. Sigmoid or Soft step
    """
    return 1.0 / (1.0 + exp(-x))


def logistic_derivative(x):
    return logistic(x) * (1.0 - logistic(x))


def reLU(x):
    return 0 if x <= 0 else x


def reLU_derivative(x):
    return 0 if x <= 0 else 1


def softPlus(x):
    return log(1.0 + exp(x))


def softPlus_derivative(x):
    return 1.0 / (1.0 + exp(-x))
