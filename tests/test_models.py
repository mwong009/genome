import numpy as np
import theano.tensor as TT
from genome import models


def test_BaseModel():
    input = np.random.random(size=(4, 10))
    output = np.random.randint(low=0, high=7, size=(4, 10))
    model = models.BaseModel(input, output)
    y = np.random.randint(low=0, high=7, size=(4, 10))

    model.negative_log_likelihood(y)


def test_MultinomialLogit():
    x = TT.matrix('x')
    y = TT.vector('y')
    model = models.MultinomialLogit(
        input=x, output=y, n_vars=10, n_choices=10, beta=None, asc=None)
