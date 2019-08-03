from math import *


def tanH(x):
    return tanh(x)


def tanH_derivative(x):
    return 1 - (x*x)


def reLU(x):
    return 0 if x <= 0 else x


def reLU_derivative(x):
    return 0 if x <= 0 else 1


def softPlus(x) :
    return log(1 + exp(x))


def softPlus_derivative(x) :
    return 1 / (1 + exp(-x))
