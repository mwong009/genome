#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from genome.models import base, logit

import numpy as np
import pandas as pd
import theano
import theano.tensor as T

FLOATX = theano.config.floatX

data = pd.read_csv('data/test_data.csv')


def test_LinearRegression():
    x_data = data[['weekend', 'trip_time', 'act_leisure']]
    y_data = data['trip_dist']

    train_x_data, valid_x_data = x_data.iloc[:70], x_data.iloc[70:]
    train_y_data, valid_y_data = y_data.iloc[:70], y_data.iloc[70:]

    train_x_shared = theano.shared(train_x_data.values.astype(FLOATX), borrow=True)
    train_y_shared = theano.shared(train_y_data.values.astype(FLOATX), borrow=True)

    valid_x_shared = theano.shared(valid_x_data.values.astype(FLOATX), borrow=True)
    valid_y_shared = theano.shared(valid_y_data.values.astype(FLOATX), borrow=True)

    # Symbolic Tensors
    x = T.matrix('x')
    y = T.vector('y')
    index = T.lscalar()

    # model config
    n_vars = x_data.shape[-1]

    model = logit.LinearRegression(input=x, n_vars=n_vars)
    mse = model.mean_squared_error(y)

    model.train_model = theano.function(
        inputs=[x, y],
        outputs=mse,
        updates=None,
        givens=None,
        allow_input_downcast=True,
        on_unused_input='ignore'
    )

    out = model.train_model(train_x_data, train_y_data)
    print(out)


def test_BaseModel():
    input = np.random.random(size=(4, 10))
    output = np.random.randint(low=0, high=7, size=(4, 10))
    model = base.BaseModel(input)
    y = np.random.randint(low=0, high=7, size=(4, 10))


def test_MultinomialLogit():
    x = T.matrix('x')
    model = logit.MultinomialLogit(
        input=x, n_vars=10, n_choices=10, beta=None, asc=None)


def test_MLP():
    x = T.matrix('x')
    y = T.vector('y')
    model = logit.MLP(
        input=x, n_in=5, n_out=3,
        layers=[(5, 10, T.nnet.sigmoid), (10, 3, T.nnet.softmax)])
