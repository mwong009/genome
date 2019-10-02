# -*- coding: utf-8 -*-
"""Optimizers for model estimation

Genome implements various optimizers for model estimation.

If you would like to add additional optimization, refer to the guide on
:ref:`extending`.
"""
import numpy as np
import theano
import theano.tensor as T

FLOATX = theano.config.floatX


class BaseOpt:
    def __init__(self, params, learning_rate, consider_constants):
        assert isinstance(params, list)
        assert learning_rate > 0

        self.learning_rate = learning_rate
        self.consider_constants = consider_constants

    def set_learning_rate(self, lr):
        self.learning_rate = lr

    def set_consider_constants(self, consider_constants):
        self.consider_constants = consider_constants


class SGD(BaseOpt):
    """
    Stocastic gradient descent - An iterative method for optimizing an objective
    function

    Computes the gradient of the cost function w.r.t. the parameters :math:`\\theta`
    for each batch dataset:

    .. math::

        \\theta^{new} = \\theta^{old} - \\eta\\cdot\\nabla_{\\theta}J(x,y)

    Args:
        params (list): A list of ``TensorSharedVariable`` model parameters
    """
    def __init__(self, params, learning_rate=0.1, consider_constants=None):
        super().__init__(params, learning_rate, consider_constants)

    def update(self, cost, params):
        grads = []
        for param in params:
            grad = T.grad(cost, param, self.consider_constants)
            grads.append(grad)

        updates = []
        for param, grad in zip(params, grads):
            update = param - self.learning_rate * grad
            updates.append((param, update))

        return updates


class MomentumSGD(BaseOpt):
    def __init__(self, params, learning_rate=0.1, momentum=0.9,
                 consider_constants=None, use_nesterov=False):
        super().__init__(params, learning_rate, consider_constants)

        assert (momentum < 1.) & (momentum > 0)
        self.momentum = momentum
        self.use_nestrov = use_nesterov

        self.velocity = []
        for param in params:
            velocity = theano.shared(
                value=np.zeros_like(param.get_value(), dtype=FLOATX))
            self.velocity.append(velocity)

    def set_momentum(self, momentum):
        assert (momentum < 1.) & (momentum > 0)
        self.momentum = momentum

    def update(self, cost, params):
        grads = []
        for param in params:
            grad = T.grad(cost, param, self.consider_constants)
            grads.append(grad)

        velocity = self.velocity

        updates = []
        for v, param, grad in zip(velocity, params, grads):
            v_t = self.learning_rate * grad + self.momentum * v
            updates.append((v, v_t))

            if self.use_nestrov:
                update = param - (self.learning_rate * grad + self.momentum * v_t)
            else:
                update = param - v_t
            updates.append((param, update))

        return updates


class Adagrad(BaseOpt):
    def __init__(self, params, learning_rate=1., consider_constants=None):
        super().__init__(params, learning_rate, consider_constants)

        self.velocity = []
        for param in params:
            velocity = theano.shared(
                value=np.zeros_like(param.get_value(), dtype=FLOATX))
            self.velocity.append(velocity)

    def update(self, cost, params, epsilon=1e-6):
        grads = []
        for param in params:
            grad = T.grad(cost, param, self.consider_constants)
            grads.append(grad)

        velocity = self.velocity

        updates = []
        for v, param, grad in zip(velocity, params, grads):
            v_t = v + grad ** 2
            updates.append((v, v_t))

            update = param - self.learning_rate * grad / T.sqrt(v_t + epsilon)
            updates.append((param, update))

        return updates


class RMSProp(BaseOpt):
    def __init__(self, params, learning_rate=0.1, rho=0.9, consider_constants=None):
        super().__init__(params, learning_rate, consider_constants)
        self.rho = rho
        assert (self.rho < 1.) & (self.rho > 0.)

        self.velocity = []
        for param in params:
            velocity = theano.shared(
                value=np.zeros_like(param.get_value(), dtype=FLOATX))
            self.velocity.append(velocity)

    def update(self, cost, params, epsilon=1e-6):
        grads = []
        for param in params:
            grad = T.grad(cost, param, self.consider_constants)
            grads.append(grad)

        velocity = self.velocity

        updates = []
        for v, param, grad in zip(velocity, params, grads):
            v_t = self.rho * v + (1. - self.rho) * grad ** 2
            updates.append((v, v_t))

            update = param - self.learning_rate * grad / T.sqrt(v_t + epsilon)
            updates.append((param, update))

        return updates


class Adadelta(BaseOpt):
    def __init__(self, params, learning_rate=1.0, rho=0.95, consider_constants=None):
        super().__init__(params, learning_rate, consider_constants)
        self.rho = rho
        assert (self.rho < 1.) & (self.rho > 0.)

        self.velocity = []
        self.delta = []
        for param in params:
            velocity = theano.shared(
                value=np.zeros_like(param.get_value(), dtype=FLOATX))
            self.velocity.append(velocity)

            delta = theano.shared(
                value=np.zeros_like(param.get_value(), dtype=FLOATX))
            self.delta.append(delta)

    def update(self, cost, params, epsilon=1e-6):
        grads = []
        for param in params:
            grad = T.grad(cost, param, self.consider_constants)
            grads.append(grad)

        velocity = self.velocity
        delta = self.delta

        updates = []
        for v, d, param, grad in zip(velocity, delta, params, grads):
            v_t = self.rho * v + (1. - self.rho) * grad ** 2
            updates.append((v, v_t))

            update = param - self.learning_rate * (
                grad * T.sqrt(d + epsilon) / T.sqrt(v_t + epsilon))
            updates.append((param, update))

            d_t = self.rho * d + (1. - self.rho) * (
                grad * T.sqrt(d + epsilon) / T.sqrt(v_t + epsilon)) ** 2
            updates.append((d, d_t))

        return updates


class Adam(BaseOpt):
    def __init__(self, params, learning_rate=1., beta_1=0.9, beta_2=0.999,
                 consider_constants=None):
        super().__init__(params, learning_rate, consider_constants)

        self.velocity = []
        self.delta = []
        for param in params:
            velocity = theano.shared(
                value=np.zeros_like(param.get_value(), dtype=FLOATX))
            self.velocity.append(velocity)

            delta = theano.shared(
                value=np.zeros_like(param.get_value(), dtype=FLOATX))
            self.delta.append(delta)

        self.step = theano.shared(value=np.zeros((1,), dtype=FLOATX))
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def update(self, cost, params, epsilon=1e-6):
        grads = []
        for param in params:
            grad = T.grad(cost, param, self.consider_constants)
            grads.append(grad)

        velocity = self.velocity
        delta = self.delta

        updates = []
        t = self.step + T.constant(1.)
        updates.append((self.step, t))
        for v, d, param, grad in zip(velocity, delta, params, grads):
            v_t = self.beta_1 * v + (1. - self.beta_1) * grad
            updates.append((v, v_t))

            d_t = self.beta_2 * d + (1. - self.beta_2) * grad ** 2
            updates.append((d, d_t))

            v_hat = v_t / (1. - self.beta_1 ** t)
            d_hat = d_t / (1. - self.beta_2 ** t)
            update = param - self.learning_rate * v_hat / (T.sqrt(d_hat) + epsilon)
            updates.append((param, update))

        return updates
