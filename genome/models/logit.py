"""
Model constructors.

"""
from genome.models import base, core

import numpy as np
# import theano
# import theano.tensor as T

FLOATX = theano.config.floatX


class LinearRegression(base.BaseModel):
    """
    Creates a Linear Regression model

    This class creates a generic linear regression model: y = Wx + b.

    Args:
        input (:obj:`TensorVariable`): Symbolic variable that describes
            the input.
        n_vars (int): Number of input variables.
        W (:obj:`TensorSharedVariable`, optional): Slope parameters.
        b (:obj:`TensorSharedVariable`, optional): Intercept parameter.

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
        """
        Returns a float representing the mean squared error (MSE)

        Args:
            y (:obj:`TensorVariable`): corresponds to the ground-truth
                value of the dependent variable
        Returns:
            The MSE of the linear model

        """
        # check if y is of the correct datatype
        if y.dtype.startswith('float'):
            return T.mean(T.sqr(y - self.output[..., 0]))
        else:
            raise NotImplementedError()


class MLP(base.MultiLayerModel):
    """
    Creates the Multilayer perceptron model

    Args:
        input (:obj:`TensorVariable`): Symbolic variable that describes
            the input.
        n_in (int): Number of input nodes.
        n_out (int): Number of output nodes.
        layers (``[(int, int, None), (int, int, None),...]``): A list of tuples that
            defines the number of input connection, output connection and layer
            activation function:
            ``[(n_in, n_hidden, activation),... (n_hidden, n_out, activation)]``.

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

            if self.hidden_layers[n-1].activation == T.nnet.softmax:
                self.output_pred = T.argmax(self.output, axis=1)


class MultinomialLogit(base.BaseModel):
    """
    Creates the standard Multinomial Logit model

    Args:
        input (:obj:`TensorVariable`): symbolic variable that describes
            the input
        n_vars (int): number of input variables
        n_choices (int): number of choice alternatives
        beta (:obj:`TensorSharedVariable`, optional): beta parameters
        asc (:obj:`TensorSharedVariable`, optional): alternative specific constants

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

    def negative_log_likelihood(self, y, mean=True):
        """
        Returns the mean or sum of the negative log likelihood

        Args:
            y (:obj:`TensorVariable`): symbolic variable that describe the
                ground-truth
            mean (bool): checks whether to average `True` the negative log-likelihood

        Returns:
            The negative log-likelihood of the mini-batch
        """
        # y.shape[0] is the mini-batch size
        nll = -T.sum(T.log(self.output)[T.arange(y.shape[0]), y])
        if not mean:
            return nll
        else:
            return nll / y.shape[0]

    def errors(self, y):
        """
        Returns a float representing the errors in the minibatch

        Args:
            y (TensorVariable): corresponds to the ground-truth vector
                that gives the correct response

        Returns:
            The mean error rate of predictions
        """
        # check if y has same dimension of y_pred
        if y.ndim != self.output.ndim:
            raise TypeError('y should have the same shape as self.output_pred',
                            ('y', y.type, 'output_pred', self.output_pred.type))

        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.output_pred, y))
        else:
            raise NotImplementedError()

    def hessians(self, y):
        """
        Return a list of hessians w.r.t. to the model parameters

        Args:
            y (:obj:`TensorSharedVariable`): corresponds to the ground-truth vector
                that gives the correct response

        Returns:
            A list of hessian matrix corresponding to each parameter
        """
        hessian_matrix = []
        for param in self.params:
            _shp = param.shape
            cost_function = self.negative_log_likelihood(y)
            _hessian = T.hessian(cost_function, param, disconnected_inputs='ignore')
            hessian_matrix.append(_hessian.reshape(_shp))

        return hessian_matrix


class ResLogit(MultinomialLogit, base.ResNetModel):
    """
    Creates the ResLogit model

    Args:
        input (:obj:`TensorVariable`): symbolic variable that describes
            the input
        n_vars (int): number of input variables
        n_choices (int): number of choice alternatives
        n_layers (int): Number of residual layers.
        beta (:obj:`TensorSharedVariable`, optional): beta parameters
        asc (:obj:`TensorSharedVariable`, optional): alternative specific constants

    """
    def __init__(self, input, n_vars, n_choices, n_layers, beta=None, asc=None):
        MultinomialLogit.__init__(self, input, n_vars, n_choices, beta, asc)
        base.ResNetModel.__init__(self, n_layers)

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
