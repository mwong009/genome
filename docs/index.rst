=================
Welcome to Genome
=================

Genome is a Python library that allows you to develop discrete choice models with machine learning and neural network tools.
The goal is to combine cutting edge machine learning techniques with discrete
choice analysis to allow better and faster model developments.
This software is specifically built for estimation of travel behaviour data
sets.
It is built using the `Theano deep learning library <https://github.com/Theano/Theano/>`_.

Genome primary interface is a basic command-line script based Python software.
Using a script allows users to describe a model 'recipe'.
The core of Genome uses the Theano engine to train neural network models.
It is adapted to work on more conventional style discrete choice models and
econometric analysis.
It allows extremely fast computational speedup using tensor based computations.
This software can be configured to automatically estimate both discrete and
continuous output simultaneously.

.. note::
    We are currently working towards a stable beta release.
    Until it is ready, Genome is only suitable for developers who can help to find bugs and issues.
    This software should **not** be used for live production.

News
====

* Genome version 0.1.0 (2019/05/03)

Contents
========

.. toctree::
    :maxdepth: 2

    about
    news
    install
    tutorials/tutorials
    development/development
    contact
    releasenotes
