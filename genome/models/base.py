"""
Base classes for model types.

"""


class BaseModel:
    """
    The BaseModel class creates the backbone of the other models

    Common methods shared between model types are defined here

    Args:
        input (:obj:`theano.tensor.TensorVariable`): symbolic variable that describes
            the input

    """
    def __init__(self, input):
        self.input = input


class ResNetModel(BaseModel):
    """
    Base class for Residual Network models

    Args:
        input (:obj:`theano.tensor.TensorVariable`): symbolic variable that describes
            the input
        n_in (int): Number of input nodes.
        n_out (int): Number of output nodes.
        n_layers (int): A list of tuples that defines the number of
            input connection, output connection and layer activation function:
            `[(n_in, n_hidden, activation),... (n_hidden, n_out, activation)]`.

    """
    def __init__(self, n_layers):
        super().__init__(input)

        self.n_layers = n_layers
        assert self.n_layers >= 2

        self.resnet_layers = []
        self.params = []
        self.params_m = []


class MultiLayerModel(BaseModel):
    """
    Base class for multi-layered models

    Args:
        input (:obj:`theano.tensor.TensorVariable`): symbolic variable that describes
            the input
        n_in (int): Number of input nodes.
        n_out (int): Number of output nodes.
        layers ([(int, int),...]): A list of tuples that defines the number of
            input connection, output connection and layer activation function:
            `[(n_in, n_hidden, activation),... (n_hidden, n_out, activation)]`.

    """
    def __init__(self, input, n_in, n_out, layers):
        super().__init__(input)

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
