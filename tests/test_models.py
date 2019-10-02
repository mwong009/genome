#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test script for models

Invokes the models module, creates MLP, MNL, ResLogit, Regression
Args:
    data (pandas.DataFrame): 100 row sample of the MTLTrajet dataset
    x_data: (100, 3) :class:``pandas.DataFrame`` object
    y_data: (100, 1) :class:``pandas.DataFrame`` object
    train_x: 70% data slice
    train_y: 30% data slice
    n_vars (int): number of x variables used (3)
"""
from genome.models import base, linear, logit, dnn

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

    assert train_x_data.ndim == train_y_data.ndim

    model = linear.LinearRegression(input=x, n_vars=n_vars)
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
    print('MLP: mse {0:.4f}'.format(out))


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

    assert train_x_data.ndim == train_y_data.ndim

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
    print('MLP: nll {0:.4f}'.format(out))

    out = model.validate_model(train_x_data, train_y_data)
    print('MLP: error {0:.4f}%'.format(out * 100))


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

    assert train_x_data.ndim == train_y_data.ndim

    layers = [
        (5, 10, T.nnet.sigmoid),
        (10, 10, T.nnet.sigmoid),
        (10, 7, T.nnet.softmax)
    ]

    n_layers = len(layers)

    model = dnn.MLP(input=x, n_in=n_vars, n_out=7, layers=layers)

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
    print('MLP ({0:d}): nll {1:.4f}'.format(n_layers, out))

    out = model.validate_model(train_x_data, train_y_data)
    print('MLP ({0:d}): error {1:.4f}%'.format(n_layers, out * 100))


def test_ResLogit():
    x_data = data[['weekend', 'hour_8_10', 'trip_time', 'trip_aspeed', 'act_home']]
    y_data = data[['mode']] - 1

    y = T.imatrix('y')

    # model config
    n_vars = x_data.shape[-1]
    train_x_data, valid_x_data = x_data.iloc[:70], x_data.iloc[70:]
    train_y_data, valid_y_data = y_data.iloc[:70], y_data.iloc[70:]

    assert train_x_data.ndim == train_y_data.ndim

    n_layers = 3

    model = dnn.ResLogit(input=x, n_vars=n_vars, n_choices=7, n_layers=n_layers)

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
    print('ResLogit ({0:d}): nll {1:.4f}'.format(n_layers, out))

    out = model.validate_model(train_x_data, train_y_data)
    print('ResLogit ({0:d}): error {1:.4f}%'.format(n_layers, out * 100))
