#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from genome.models import base, logit

import numpy as np
import pandas as pd
import theano
import theano.tensor as T

FLOATX = theano.config.floatX

data = pd.read_csv('data/test_data.csv')

# Symbolic Tensors
x = T.matrix('x')
y = T.matrix('y')
index = T.lscalar()


def test_BaseModel():
    input = np.random.random(size=(4, 10))
    output = np.random.randint(low=0, high=7, size=(4, 10))
    model = base.BaseModel(input)
    y = np.random.randint(low=0, high=7, size=(4, 10))


def test_LinearRegression():
    x_data = data[['weekend', 'trip_time', 'act_leisure']]
    y_data = data[['trip_dist']]

    # model config
    n_vars = x_data.shape[-1]
    train_x_data, valid_x_data = x_data.iloc[:70], x_data.iloc[70:]
    train_y_data, valid_y_data = y_data.iloc[:70], y_data.iloc[70:]

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
    print('LinearRegression: mse', out)


def test_MultinomialLogit():
    x_data = data[['weekend', 'hour_8_10', 'trip_time', 'trip_aspeed', 'act_home']]
    y_data = data[['mode']] - 1

    # For Logistic Regression, y-output must be an integer TensorType
    # T.imatrix() or T.ivector()
    y = T.imatrix('y')

    # model config
    n_vars = x_data.shape[-1]
    train_x_data, valid_x_data = x_data.iloc[:70], x_data.iloc[70:]
    train_y_data, valid_y_data = y_data.iloc[:70], y_data.iloc[70:]

    model = logit.MultinomialLogit(input=x, n_vars=n_vars, n_choices=7)
    nll = model.negative_log_likelihood(y)
    errors = model.errors(y)

    model.train_model = theano.function(
        inputs=[x, y],
        outputs=nll,
        updates=None,
        givens=None,
        allow_input_downcast=True,
        on_unused_input='ignore'
    )

    model.validate_model = theano.function(
        inputs=[x, y],
        outputs=errors,
        updates=None,
        givens=None,
        allow_input_downcast=True,
        on_unused_input='ignore'
    )

    out = model.train_model(train_x_data, train_y_data)
    print('MultinomialLogit: nll', out)

    out = model.validate_model(train_x_data, train_y_data)
    print('MultinomialLogit: error', '{0:2.2f}%%'.format(out * 100))


def test_MLP():
    x_data = data[['weekend', 'hour_8_10', 'trip_time', 'trip_aspeed', 'act_home']]
    y_data = data[['mode']] - 1

    # For Logistic Regression, y-output must be an integer TensorType
    # T.imatrix() or T.ivector()
    y = T.imatrix('y')

    # model config
    n_vars = x_data.shape[-1]
    train_x_data, valid_x_data = x_data.iloc[:70], x_data.iloc[70:]
    train_y_data, valid_y_data = y_data.iloc[:70], y_data.iloc[70:]

    layers = [
        (5, 10, T.nnet.sigmoid),
        (10, 10, T.nnet.sigmoid),
        (10, 7, T.nnet.softmax)
    ]
    model = logit.MLP(input=x, n_in=n_vars, n_out=7, layers=layers)

    nll = model.negative_log_likelihood(y)
    errors = model.errors(y)

    model.train_model = theano.function(
        inputs=[x, y],
        outputs=nll,
        updates=None,
        givens=None,
        allow_input_downcast=True,
        on_unused_input='ignore'
    )

    model.validate_model = theano.function(
        inputs=[x, y],
        outputs=errors,
        updates=None,
        givens=None,
        allow_input_downcast=True,
        on_unused_input='ignore'
    )

    out = model.train_model(train_x_data, train_y_data)
    print('MLP (3): nll', out)

    out = model.validate_model(train_x_data, train_y_data)
    print('MLP (3): error', '{0:2.2f}%%'.format(out * 100))
