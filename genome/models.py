"""Model constructors"""

import numpy as np
import theano
import theano.tensor as T


FLOATX = theano.config.floatX


class BaseModel:
    """The BaseModel class creates the backbone of the other models

    Common methods shared between model types are defined here

    Args:
        input (theano.tensor.TensorVariable): symbolic variable that describes the
            input
    """
    def __init__(self, input):
        self.input = input


class MultiLayerModel:
    """Base class for multi-layered models

    Args:
        input (theano.tensor.TensorVariable): symbolic variable that describes the
            input
        n_in (int): Number of input nodes.
        n_out (int): Number of output nodes.
        layers ([(int, int),...]): A list of tuples that defines the number of
            input connection, output connection and layer activation function:
            `[(n_in, n_hidden, activation),... (n_hidden, n_out, activation)]`.
    """
    def __init__(self, input, n_in, n_out, layers):
        self.input = input

        assert isinstance(layers, list)
        self.n_layers = len(layers)

        for n, layer in enumerate(layers):
            assert (len(layer) >= 2 & len(layer) <= 3)
            if len(layer) == 2:
                layers[n] = layers[n] + (None,)
            else:
                assert callable(layer[2])
            if n == 0:
                assert layer[0] == n_in
            else:
                assert layer[0] == layers[n-1][1]
            if n == self.n_layers-1:
                assert layer[1] == n_out
        self.layers = layers

        self.hidden_layers = []
        self.params = []
        self.params_m = []


class LinearRegression(object):
    """Creates a Linear Regression model

    This class creates a generic linear regression model: y = Wx + b.

    Args:
        input (theano.tensor.TensorVariable): Symbolic variable that describes the
            input.
        n_vars (int): Number of input variables.
        W (TensorSharedVariable, optional): Slope parameters.
        b (TensorSharedVariable, optional): Intercept parameter.
    """
    def __init__(self, input, n_vars, W=None, b=None):
        self.input = input

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
        """Returns a float representing the mean squared error (MSE)

        Args:
            y (theano.tensor.TensorVariable): corresponds to the ground-truth value
                of the dependent variable
        Returns:
            TensorSharedVariable: The MSE of the linear model
        """
        # check if y is of the correct datatype
        if y.dtype.startswith('float'):
            return T.mean(T.sqr(y - self.output[..., 0]))
        else:
            raise NotImplementedError()


class HiddenLayer(object):
    """Hidden Layer class object

    This abstract class is used for a wrapper for intermediate layers. Common activation
    function used are `T.nnet.sigmoid`, `T.nnet.softplus`, `T.nnet.relu`, `T.tanh`.

    Args:
        input (theano.tensor.TensorVariable): Symbolic variable that describes the
            input.
        n_in (int): Number of input nodes.
        n_out (int): Number of output nodes.
        W (TensorSharedVariable, optional): Weight parameters.
        bias (TensorSharedVariable, optional): Bias parameters.
        activation (function, optional): Activation function, defaults to linear
            regression if undefined.
    """
    def __init__(self, input, n_in, n_out, W=None, bias=None, activation=None):
        self.input = input
        self.activation = activation

        if W is None:
            init = np.zeros((n_in, n_out), dtype=FLOATX)
            _W = theano.shared(value=init.flatten(), name='_W', borrow=True)
            W = _W.reshape((n_in, n_out))
            W.name = 'W'
        self._W = _W
        self.W = W

        if bias is None:
            init = np.zeros((n_out,), dtype=FLOATX)
            bias = theano.shared(value=init, name='bias', borrow=True)
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


class MLP(MultiLayerModel):
    """Creates the Multilayer perceptron model

    Args:
        input (theano.tensor.TensorVariable): Symbolic variable that describes the
            input.
        n_in (int): Number of input nodes.
        n_out (int): Number of output nodes.
        layers ([(int, int),...]): A list of tuples that defines the number of
            input connection, output connection and layer activation function:
            `[(n_in, n_hidden, activation),... (n_hidden, n_out, activation)]`.
    """
    def __init__(self, input, n_in, n_out, layers):
        """Initializes the MLP model"""
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
                input=layer_input, n_in=layer_n_in, n_out=layer_n_out,
                activation=layer_activaton
            )

            self.hidden_layers.append(hidden_layer)
            self.params.extend(hidden_layer.params)
            self.params_m.extend(hidden_layer.params_m)

            self.output = self.hidden_layers[-1].output

            if self.hidden_layers[n-1].activation == T.nnet.softmax:
                self.output_pred = T.argmax(self.output, axis=1)


class ResNet(MLP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass


class MultinomialLogit(BaseModel):
    """Creates the standard Multinomial Logit model

    Args:
        input (theano.tensor.TensorVariable): symbolic variable that describes the
            input
        n_vars (int): number of input variables
        n_choices (int): number of choice alternatives
        beta (TensorSharedVariable, optional): beta parameters
        asc (TensorSharedVariable, optional): alternative specific constants
    """
    def __init__(self, input, n_vars, n_choices, beta=None, asc=None):
        """Initializes the Logit class"""
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
        """Returns the mean or sum of the negative log likelihood

        Args:
            y (theano.tensor.TensorVariable): symbolic variable that describe the
                ground-truth
            mean (bool): checks whether to average `True` the negative log-likelihood
        Returns:
            TensorSharedVariable: negative log-likelihood of the mini-batch
        """
        # y.shape[0] is the mini-batch size
        nll = -T.sum(T.log(self.output)[T.arange(y.shape[0]), y])
        if not mean:
            return nll
        else:
            return nll / y.shape[0]

    def errors(self, y):
        """Returns a float representing the errors in the minibatch

        Args:
            y (theano.tensor.TensorVariable): corresponds to the ground-truth vector
                that gives the correct response
        Returns:
            TensorSharedVariable: The mean error rate of predictions
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
        """Return a list of hessians w.r.t. to the model parameters

        Args:
            y (TensorSharedVariable): corresponds to the ground-truth vector that gives
                the correct response
        Returns:
            list(TensorSharedVariable): a list of hessian matrix corresponding to each
                parameter
        """
        hessian_matrix = []
        for param in self.params:
            _shp = param.shape
            cost_function = self.negative_log_likelihood(y)
            _hessian = T.hessian(cost_function, param, disconnected_inputs='ignore')
            hessian_matrix.append(_hessian.reshape(_shp))

        return hessian_matrix
