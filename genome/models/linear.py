# -*- coding: utf-8 -*-
"""A simple Linear regression model"""
from genome.models import base
from genome.models import functions as fn

import numpy as np
import theano
import theano.tensor as T

FLOATX = theano.config.floatX


class LinearRegression(base.BaseModel):
    """
    Initializes a Linear Regression model

    This class creates a generic linear regression model: y = Wx + b.

    Args:
        input: A symbolic ``Tensor`` input
        n_vars (int): Number of input variables
        W: Slope parameters - A ``TensorSharedVariable`` (optional)
        b: Intercept parameter - A ``TensorSharedVariable`` (optional)

    """
    def __init__(self, input, n_vars, W=None, b=None):
        super().__init__(input)

        if W is None:
            init = np.zeros((n_vars, 1), dtype=FLOATX)
            W = theano.shared(value=init, name='W', borrow=True)
        self.W = W

        if b is None:
            init = np.zeros((1,), dtype=FLOATX)
            b = theano.shared(value=init, name='b', borrow=True)
        self.b = b

        self.params = [self.W, self.b]
        self.params_m = [None, None]

        self.output = T.dot(input, self.W) + self.b

    def mean_squared_error(self, y):
        """see: :func:`~genome.models.functions.mean_squared_error`"""
        return fn.mean_squared_error(self, y)
