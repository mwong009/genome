# -*- coding: utf-8 -*-
"""
Test function description:

Invokes the optimizers module on a linear regression model

Args:
    data (pandas.DataFrame): 100 row sample of the MTLTrajet dataset
    x_data: (100, 3) :class:``pandas.DataFrame`` object
    y_data: (100, 1) :class:``pandas.DataFrame`` object
    train_x: 70% data slice
    train_y: 30% data slice
    n_vars (int): number of x variables used (3)

"""
from genome.optimizers import *
from genome.models import logit

import theano
import theano.tensor as T
import pandas as pd

data = pd.read_csv('data/test_data.csv')
x_data = data[['weekend', 'trip_time', 'act_leisure']]
y_data = data[['trip_dist']]

train_x_data, valid_x_data = x_data.iloc[:70], x_data.iloc[70:]
train_y_data, valid_y_data = y_data.iloc[:70], y_data.iloc[70:]

assert train_x_data.ndim == train_y_data.ndim

n_vars = x_data.shape[-1]

x = T.matrix('x')
y = T.matrix('y')


def test_sgd():
    model = logit.LinearRegression(input=x, n_vars=n_vars)
    sgd = SGD(model.params)
    rmse = T.sqrt(model.mean_squared_error(y))
    update = sgd.update(rmse, model.params)

    test_function = theano.function(
        inputs=[x, y],
        outputs=rmse,
        updates=update,
        allow_input_downcast=True,
    )

    for i in range(100):
        batch_cost = test_function(train_x_data.values, train_y_data.values)

    print('SGD:', batch_cost)


def test_momentumSGD():
    model = logit.LinearRegression(input=x, n_vars=n_vars)
    momsgd = MomentumSGD(model.params)
    rmse = T.sqrt(model.mean_squared_error(y))
    update = momsgd.update(rmse, model.params)

    test_function = theano.function(
        inputs=[x, y],
        outputs=rmse,
        updates=update,
        allow_input_downcast=True,
    )

    for i in range(100):
        batch_cost = test_function(train_x_data.values, train_y_data.values)

    print('MomentumSGD:', batch_cost)


def test_nesterov_momentumSGD():
    model = logit.LinearRegression(input=x, n_vars=n_vars)
    nesterov = MomentumSGD(model.params, use_nesterov=True)
    rmse = T.sqrt(model.mean_squared_error(y))
    update = nesterov.update(rmse, model.params)

    test_function = theano.function(
        inputs=[x, y],
        outputs=rmse,
        updates=update,
        allow_input_downcast=True,
    )

    for i in range(100):
        batch_cost = test_function(train_x_data.values, train_y_data.values)

    print('Nesterov:', batch_cost)


def test_Adagrad():
    model = logit.LinearRegression(input=x, n_vars=n_vars)
    adagrad_sgd = Adagrad(model.params)
    rmse = T.sqrt(model.mean_squared_error(y))
    update = adagrad_sgd.update(rmse, model.params)

    test_function = theano.function(
        inputs=[x, y],
        outputs=rmse,
        updates=update,
        allow_input_downcast=True,
    )

    for i in range(100):
        batch_cost = test_function(train_x_data.values, train_y_data.values)

    print('Adagrad:', batch_cost)


def test_RMSProp():
    model = logit.LinearRegression(input=x, n_vars=n_vars)
    rmsprop_sgd = RMSProp(model.params)
    rmse = T.sqrt(model.mean_squared_error(y))
    update = rmsprop_sgd.update(rmse, model.params)

    test_function = theano.function(
        inputs=[x, y],
        outputs=rmse,
        updates=update,
        allow_input_downcast=True,
    )

    for i in range(100):
        batch_cost = test_function(train_x_data.values, train_y_data.values)

    print('RMSProp:', batch_cost)


def test_Adadelta():
    model = logit.LinearRegression(input=x, n_vars=n_vars)
    adadelta_sgd = Adadelta(model.params)
    rmse = T.sqrt(model.mean_squared_error(y))
    update = adadelta_sgd.update(rmse, model.params)

    test_function = theano.function(
        inputs=[x, y],
        outputs=rmse,
        updates=update,
        allow_input_downcast=True,
    )

    for i in range(100):
        batch_cost = test_function(train_x_data.values, train_y_data.values)

    print('Adadelta:', batch_cost)


def test_Adam():
    model = logit.LinearRegression(input=x, n_vars=n_vars)
    adam_sgd = Adam(model.params)
    rmse = T.sqrt(model.mean_squared_error(y))
    update = adam_sgd.update(rmse, model.params)

    test_function = theano.function(
        inputs=[x, y],
        outputs=rmse,
        updates=update,
        allow_input_downcast=True,
    )

    for i in range(100):
        batch_cost = test_function(train_x_data.values, train_y_data.values)

    print('Adam:', batch_cost)
