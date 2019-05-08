.. _about:

=====
About
=====

Genome is a Python library that allows you to develop discrete choice models
with machine learning and neural network tools.
The goal is to combine cutting edge machine learning techniques with discrete
choice analysis to allow better and faster model developments.
This software is specifically built for estimation of travel behaviour data
sets.
It is built on the It is built on the `Theano deep learning library <https://github.com/Theano/Theano/>`_.

Genome primary interface is a basic command-line script based Python software.
Using a script allows users to describe a model 'recipe'.
The core of Genome uses the Theano engine to train neural network models.
It is adapted to work on more conventional style discrete choice models and
econometric analysis.
It allows extremely fast computational speedup using tensor based computations.
This software can be configured to automatically estimate both discrete and
continuous output simultaneously.

.. topic:: Features

    * Specify multiple layers for a deep learning based discrete choice model
    * Deep learning optimization algorithms (SGD, Adam, N-Momentum, Adagrad)
    * Supports standard MNL model with reference parameters
    * Generates estimated parameter tables with standard error calculations
    * Loglikelihood tracking


.. topic:: Under Development

    * Generative model implementation
    * Residual networks
    * Model saving and reloading
