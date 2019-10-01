"""Model constructors"""

from genome.models import base, core
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
        return fn.mean_squared_error(self, y)


class MLP(base.MultiLayerModel):
    """
    Initializes a Multilayer perceptron model

    Args:
        input: A symbolic ``Tensor`` input
        n_in (int): Number of input variables
        n_out (int): Number of output variables
        layers (list): A list of tuples of ``TensorSharedVariable`` that defines the
            number of input connection, output connection and layer activation function,
            for example: ``[(n_in, n_hidden, activation),... (n_hidden, n_out,
            activation)]``

    Example:
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

            hidden_layer = core.HiddenLayer(
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
        return fn.negative_log_likelihood(self, y)

    def errors(self, y):
        return fn.errors(self, y)


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
        return fn.negative_log_likelihood(self, y)

    def errors(self, y):
        return fn.errors(self, y)


class ResLogit(MultinomialLogit):
    """
    Initializes a ResLogit model

    The ResLogit model consists of a series of residual blocks that captures the
    heterogeneity in the data. It is an extension of the :class:`MultinomialLogit`
    class.

    Args:
        input: A symbolic ``Tensor`` input
        n_vars (int): Number of input variables
        n_choices (int): Number of choice alternatives
        n_layers (int): Number of residual layers
        beta: :math:`\\beta` parameters - A ``TensorSharedVariable`` (optional)
        asc: Alternative Specific Constants - A ``TensorSharedVariable`` (optional)

    """
    def __init__(self, input, n_vars, n_choices, n_layers, beta=None, asc=None):
        MultinomialLogit.__init__(self, input, n_vars, n_choices, beta, asc)

        self.n_layers = n_layers
        assert self.n_layers >= 2
        self.resnet_layers = []

        resnet_input = T.dot(self.input, self.beta)

        for n, layer in range(self.n_layers):
            if n == 0:
                layer_input = resnet_input
            else:
                layer_input = self.resnet_layers[-1].output

            resnet_layer = core.ResNetLayer(
                input=layer_input, size=(n_choices, n_choices), layer_num=n)
            self.resnet_layers.append(resnet_layer)
            self.params.extend(resnet_layer.params)
            self.params_m.extend(resnet_layer.params_m)

            # overrides Multinomial function
            self.output = T.nnet.softmax(self.resnet_layers[-1].output + self.asc)
            self.output_pred = T.argmax(self.output, axis=1)
