# -*- coding: utf-8 -*-
"""Model constructors for logit type models

The following are some of the logit type models already implemented:
- multinomial logit

.. todo::
    - latent class logit
    - ordered logit (P-logit, B-logit)
    - nested logit
    - mixed logit
"""
from genome.models import base
from genome.models import functions as fn

import numpy as np
import theano
import theano.tensor as T

FLOATX = theano.config.floatX


class MultinomialLogit(base.BaseModel):
    """
    Initializes a standard `Multinomial Logit` (MNL) model

    Args:
        input: A symbolic ``Tensor`` input
        n_vars (int): Number of input variables
        n_choices (int): Number of choice alternatives
        beta: :math:`\\beta` parameters - A ``TensorSharedVariable`` (optional)
        asc: Alternative Specific Constants - A ``TensorSharedVariable`` (optional)

    """
    def __init__(self, input, n_vars, n_choices, beta=None, asc=None):
        super().__init__(input)

        if type(n_choices) == dict:
            n_choices = len(n_choices)

        if asc is None:
            init = np.zeros((n_choices,), dtype=FLOATX)
            asc = theano.shared(value=init, name='asc', borrow=True)
        self.asc = asc

        if beta is None:
            init = np.zeros((n_vars, n_choices), dtype=FLOATX)
            _beta = theano.shared(value=init.flatten(), name='_beta', borrow=True)
            beta = _beta.reshape((n_vars, n_choices))
            beta.name = 'beta'
        self._beta = _beta
        self.beta = beta

        self.params = [self._beta, self.asc]

        self.beta_m = np.ones_like(np.zeros((n_vars, n_choices), dtype=FLOATX))
        self.beta_m[..., -1] = 0
        self.beta_m = self.beta_m.flatten()

        self.asc_m = np.ones_like(np.zeros((n_choices,), dtype=FLOATX))
        self.asc_m[..., -1] = 0

        self.params_m = [self.beta_m, self.asc_m]

        self.output = T.nnet.softmax(T.dot(input, self.beta) + self.asc)
        self.output_pred = T.argmax(self.output, axis=1)

    def negative_log_likelihood(self, y):
        """see: :func:`~genome.models.functions.negative_log_likelihood`"""
        return fn.negative_log_likelihood(self, y)

    def errors(self, y):
        """see: :func:`~genome.models.functions.errors`"""
        return fn.errors(self, y)
