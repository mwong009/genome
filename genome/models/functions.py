# -*- coding: utf-8 -*-
"""Model functions"""

import theano
import theano.tensor as T


def mean_squared_error(self, y):
    """
    Returns a `float` representing the mean squared error (MSE)

    Args:
        y: A symbolic ``Tensor`` variable of the ground-truth
    Returns:
        The MSE of the linear model
    """
    # check if y is of the correct datatype
    if y.dtype.startswith('float'):
        return T.mean(T.sqr(y - self.output))
    else:
        raise NotImplementedError()


def negative_log_likelihood(self, y, mean=True):
    """
    Returns the mean or sum of the negative log likelihood

    Args:
        y: A symbolic ``Tensor`` variable of the ground-truth
        mean (bool): checks whether to average `True` the negative log-likelihood

    Returns:
        The negative log-likelihood of the mini-batch

    .. note::
        We use `mean` so the gradient is not dependent on the size of the
        batch. set ``mean=False`` if the optimization requires dependency on the
        size of the batch.
    """
    # y.shape[0] is the mini-batch size
    nll = -T.sum(T.log(self.output)[T.arange(y.shape[0]), y[..., -1]])
    if not mean:
        return nll
    else:
        return nll / y.shape[0]


def errors(self, y):
    """
    Returns a `float` representing the errors in the minibatch

    Args:
        y: A symbolic ``Tensor`` variable of the ground-truth

    Returns:
        The mean error rate of predictions
    """
    # check if y has same dimension of y_pred
    if y.ndim != self.output.ndim:
        raise TypeError('y should have the same shape as self.output_pred',
                        ('y', y.type, 'output_pred', self.output_pred.type))

    # check if y is of the correct datatype
    if y.dtype.startswith('int'):
        return T.mean(T.neq(self.output_pred, y[..., -1]))
    else:
        raise NotImplementedError()


def hessians(self, y):
    """
    Return a list of hessians w.r.t. to the model parameters

    Args:
        y: A symbolic ``Tensor`` variable of the ground-truth

    Returns:
        A list of hessian ``TensorSharedVariable`` matrices corresponding to each
            parameter
    """
    hessian_matrix = []
    for param in self.params:
        _shp = param.shape
        cost_function = self.negative_log_likelihood(y)
        _hessian = T.hessian(cost_function, param, disconnected_inputs='ignore')
        hessian_matrix.append(_hessian.reshape(_shp))

    return hessian_matrix
