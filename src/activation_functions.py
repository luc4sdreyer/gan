from math import *


class tanH():
    def f(x):
        return tanh(x)

    def df(x):
        return 1.0 - (tanH.f(x) * tanH.f(x))


class logistic():
    """
    (a.k.a. Sigmoid or Soft step
    """
    def f(x):
        return 1.0 / (1.0 + exp(-x))

    def df(x):
        return logistic.f(x) * (1.0 - logistic.f(x))


class reLU():
    def f(x):
        return 0.0 if x <= 0 else x

    def df(x):
        return 0.0 if x <= 0 else 1.0


class softPlus():
    def f(x):
        return log(1.0 + exp(x))

    def df(x):
        return 1.0 / (1.0 + exp(-x))
