# -*- coding: utf-8 -*-
"""Deep Neural Network models"""
from genome.models import base
from genome.models import functions as fn
from genome.models.logit import MultinomialLogit

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


class MLP(base.MultiLayerModel):
    """
    Initializes a Multilayer perceptron model

    Parameters
    ----------
    input: theano.tensor.TensorVariable
        symbolic variable that describes the input
    n_in: int
        Number of input variables
    n_out:int
        Number of output variables
    layers: list of theano.shared.TensorSharedVariable
        Defines the number of input connection, output connection and layer activation
        function. Example:
        ``[(n_in, n_hidden, activation),... (n_hidden, n_out, activation)]``

    Example
    -------
    >>> x = T.matrix('x')
    >>> model = MLP(input=x, n_in=5, n_out=2, layers=[(5, 4), (4, 2)])

    """
    def __init__(self, input, n_in, n_out, layers):
        super().__init__(input, n_in, n_out, layers)

        for n, layer in enumerate(self.layers):
            layer_n_in = layer[0]
            layer_n_out = layer[1]
            layer_activaton = layer[2]
            if n == 0:
                layer_input = self.input
            else:
                layer_input = self.hidden_layers[n-1].output

            hidden_layer = HiddenLayer(
                input=layer_input, n_in=layer_n_in, n_out=layer_n_out, layer_num=n,
                activation=layer_activaton
            )

            self.hidden_layers.append(hidden_layer)
            self.params.extend(hidden_layer.params)
            self.params_m.extend(hidden_layer.params_m)

            self.output = self.hidden_layers[-1].output

        if self.hidden_layers[-1].activation == T.nnet.softmax:
            self.output_pred = T.argmax(self.output, axis=1)
        else:
            raise NotImplementedError

    def negative_log_likelihood(self, y):
        """see: :func:`~genome.models.functions.negative_log_likelihood`"""
        return fn.negative_log_likelihood(self, y)

    def errors(self, y):
        """see: :func:`~genome.models.functions.errors`"""
        return fn.errors(self, y)


class ResLogit(MultinomialLogit):
    """
    Initializes a ResLogit model

    The ResLogit model consists of a series of residual blocks that captures the
    heterogeneity in the data. It is an extension of the :class:`MultinomialLogit`
    class.

    Parameters
    ----------
    input: theano.tensor.TensorVariable
        symbolic variable that describes the input
    n_vars: int
        Number of input variables
    n_choices: int
        Number of choice alternatives
    n_layers: int
        Number of residual layers
    beta: theano.shared.TensorSharedVariable, optional
        :math:`\\beta` parameters
    asc: theano.shared.TensorSharedVariable, optional
        Alternative Specific Constants

    """
    def __init__(self, input, n_vars, n_choices, n_layers, beta=None, asc=None):
        MultinomialLogit.__init__(self, input, n_vars, n_choices, beta, asc)

        self.n_layers = n_layers
        assert self.n_layers >= 2
        self.resnet_layers = []

        resnet_input = T.dot(self.input, self.beta)

        for n in range(self.n_layers):
            if n == 0:
                layer_input = resnet_input
            else:
                layer_input = self.resnet_layers[-1].output

            residual_shape = (n_choices, n_choices)

            resnet_layer = ResNetLayer(
                input=layer_input, size=residual_shape, layer_num=n)
            self.resnet_layers.append(resnet_layer)
            self.params.extend(resnet_layer.params)
            self.params_m.extend(resnet_layer.params_m)

            # overrides Multinomial function
            self.output = T.nnet.softmax(self.resnet_layers[-1].output + self.asc)
            self.output_pred = T.argmax(self.output, axis=1)
