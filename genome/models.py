""" The core module of the project
"""


import numpy as np
import theano
import theano.tensor as TT

__license__ = "MIT"
__revision__ = "0.0.1"
__docformat__ = "reStructuredText"

FLOATX = theano.config.floatX


class BaseModel:
    """BaseModel

    The BaseModel class creates the backbone of the other models

    Args:
        input (theano.tensor.TensorVariable): symbolic variable that describes the
            input
        output (theano.tensor.TensorVariable): symbolic variable that describes the
            output
    """
    def __init__(self, input, output):
        self.input = input
        self.output = output

    def negative_log_likelihood(self, y, mean=True):
        """Returns the mean or sum of the negative log likelihood

        Args:
            y (theano.tensor.TensorVariable): symbolic variable that describe the
                output
        Returns:
            TensorSharedVariable: negative log-likelihood of the mini-batch
        """
        # y.shape[0] is the mini-batch size
        nll = -TT.sum(TT.log(self.output)[TT.arange(y.shape[0]), y])
        if not mean:
            return nll
        else:
            return nll / y.shape[0]


class LinearRegression(BaseModel):
    """Linear Regression Class"""
    def __init__(self, input, output, ):
        assert output.ndim == 2


class HiddenLayer(object):
    """Hidden Layer class object

    This abstract class is used for a wrapper for intermediate layers. Common activation
    function used are T.nnet.sigmoid, T.nnet.softplus, T.nnet.relu, T.tanh.

    Args:
        input (theano.tensor.TensorVariable): symbolic variable that describes the
            input
        n_in (int): number of input nodes
        n_out (int): number of output nodes
        W (TensorSharedVariable, optional): weight parameters
        bias (TensorSharedVariable, optional): bias parameters
        activation (function, optional): activation function, defaults to linear
        regression if undefined
    """
    def __init__(self, input, n_in, n_out, W=None, bias=None, activation=None):
        self.input = input
        self.activation = activation

        if bias is None:
            init = np.zeros((n_out,), dtype=FLOATX)
            bias = theano.shared(value=init, name='bias', borrow=True)
        self.bias = bias

        if W is None:
            init = np.zeros((n_in, n_out), dtype=FLOATX)
            _W = theano.shared(value=init.flatten(), name='_W', borrow=True)
            W = _W.reshape((n_in, n_out))
            W.name = 'W'
        self._W = _W
        self.W = W

        self.params = [self._W, self.bias]

        W_m = np.ones_like(np.zeros((n_in, n_out), dtype=FLOATX))
        W_m[..., -1] = 0
        W_m = W_m.flatten()

        bias_m = np.ones_like(np.zeros((n_out,), dtype=FLOATX))
        bias_m[..., -1] = 0

        self.params_m = [W_m, bias_m]

        lin_output = TT.dot(input, self.W) + self.bias

        if self.activation is None:
            self.output = lin_output
        else:
            assert callable(self.activation)
            self.output = self.activation(lin_output)


class MLP(object):
    def __init__(self, *args, **kwargs):
        pass


class ResNet(MLP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass


class MultinomialLogit(BaseModel):
    """Initializes the Logit class

    Args:
        input (theano.tensor.TensorVariable): symbolic variable that describes the
            input
        output (theano.tensor.TensorVariable): symbolic variable that describes the
            output
        n_vars (int): number of input variables
        n_choices (int): number of choice alternatives
        beta (TensorSharedVariable, optional): beta parameters
        asc (TensorSharedVariable, optional): alternative specific constants
    """
    def __init__(self, input, output, n_vars, n_choices, beta=None, asc=None):
        super().__init__(input, output)

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

        beta_m = np.ones_like(np.zeros((n_vars, n_choices), dtype=FLOATX))
        beta_m[..., -1] = 0
        beta_m = beta_m.flatten()

        asc_m = np.ones_like(np.zeros((n_choices,), dtype=FLOATX))
        asc_m[..., -1] = 0

        self.params_m = [beta_m, asc_m]

        self.output_prob = TT.nnet.softmax(TT.dot(input, self.beta) + self.asc)
        self.output_pred = TT.argmax(self.output_prob, axis=1)

    def errors(self, y):
        """Returns a float representing the errors in the minibatch

        Args:
            y (theano.tensor.TensorVariable): corresponds to the ground-truth vector
                that gives the correct response
        Returns:
            TensorSharedVariable: The mean error rate of predictions
        """
        # check if y has same dimension of y_pred
        if y.ndim != self.output_pred.ndim:
            raise TypeError('y should have the same shape as self.output_pred',
                            ('y', y.type, 'y_pred', self.output_pred.type))

        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            return TT.mean(TT.neq(self.output_pred, y))
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
            _hessian = TT.hessian(cost_function, param, disconnected_inputs='ignore')
            hessian_matrix.append(_hessian.reshape(_shp))

        return hessian_matrix
