# -*- coding: utf-8 -*-
"""Layer templates for creating deep neural networks"""

import numpy as np
import theano
import theano.tensor as T

FLOATX = theano.config.floatX


class HiddenLayer:
    """
    Hidden Layer class object

    This abstract class is used for a wrapper for intermediate layers. Common activation
    function used are :class:`T.nnet.sigmoid`, :class:`T.nnet.softplus`,
    :class:`T.nnet.relu`, :class:`T.tanh`.

    Args:
        input: A symbolic ``Tensor`` input
        n_in (int): Number of input variables.
        n_out (int): Number of output variables.
        layer_num (int): The n-th layer of the neural network
        W: Weight parameters - A ``TensorSharedVariable`` (optional)
        bias: Bias parameters - A ``TensorSharedVariable`` (optional)
        activation: Activation function, example :class:`T.nnet.softplus`,
            defaults to linear regression if undefined

    """
    def __init__(self, input, n_in, n_out, layer_num, W=None, bias=None,
                 activation=None):
        self.input = input
        self.activation = activation
        num = str(layer_num)

        if W is None:
            init = np.zeros((n_in, n_out), dtype=FLOATX)
            _W = theano.shared(value=init.flatten(), name='_W'+num, borrow=True)
            W = _W.reshape((n_in, n_out))
            W.name = 'W'+num
        self._W = _W
        self.W = W

        if bias is None:
            init = np.zeros((n_out,), dtype=FLOATX)
            bias = theano.shared(value=init, name='bias'+num, borrow=True)
        self.bias = bias

        self.params = [self._W, self.bias]

        self.W_m = np.ones_like(np.zeros((n_in, n_out), dtype=FLOATX))
        self.W_m[..., -1] = 0
        self.W_m = self.W_m.flatten()

        self.bias_m = np.ones_like(np.zeros((n_out,), dtype=FLOATX))
        self.bias_m[..., -1] = 0

        self.params_m = [self.W_m, self.bias_m]

        lin_output = T.dot(input, self.W) + self.bias

        if self.activation is None:
            self.output = lin_output
        else:
            assert callable(self.activation)
            self.output = self.activation(lin_output)


class ResNetLayer:
    """
    A ResNetLayer implementing a softplus activation and square matrix

    For now, we don't use reference parameters (masking) for the residual matrix.
    Reason -- no logical reason to do so.

    Args:
        input: A symbolic ``Tensor`` input
        size (tuple(int, int)): The shape of the residual correlation matrix
        layer_num (int): The n-th layer of the neural network
        W: Weight parameters - A ``TensorSharedVariable`` (optional)

    """
    def __init__(self, input, size, layer_num, W=None):
        self.input = input
        num = str(layer_num)

        if W is None:
            init = np.diag(np.ones(size[-1], dtype=FLOATX))
            _W = theano.shared(value=init.flatten(), name='_W'+num, borrow=True)
            W = _W.reshape(size)
            W.name = 'W'+num
        self._W = _W
        self.W = W

        self.params = [self._W]

        # self.W_m = np.ones_like(np.zeros(size, dtype=FLOATX))
        # self.W_m[..., -1] = 0
        # self.W_m = self.W_m.flatten()

        # self.params_m = [self.W_m]

        self.params_m = [None]

        lin_output = T.dot(self.input, self.W)
        self.output = self.input - T.nnet.softplus(lin_output)
