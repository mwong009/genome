#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import theano.tensor as T
from genome import models


def test_BaseModel():
    input = np.random.random(size=(4, 10))
    output = np.random.randint(low=0, high=7, size=(4, 10))
    model = models.BaseModel(input)
    y = np.random.randint(low=0, high=7, size=(4, 10))


def test_MultinomialLogit():
    x = T.matrix('x')
    model = models.MultinomialLogit(
        input=x, n_vars=10, n_choices=10, beta=None, asc=None)


def test_LinearRegression():
    x = T.matrix('x')
    y = T.vector('y')
    model = models.LinearRegression(input=x, n_vars=10)
    mse = model.mean_squared_error(y)


def test_MLP():
    x = T.matrix('x')
    y = T.vector('y')
    model = models.MLP(
        input=x, n_in=5, n_out=3,
        layers=[(5, 10, T.nnet.sigmoid), (10, 3, T.nnet.softmax)])
