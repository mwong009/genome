"""Genome"""
from genome.base import Environment
from genome.events import Setup

from genome.database import make_shared

import theano.tensor as T

DEFAULT_CONFIG = {
    'batch_size': 32,
}

class Genome(object):
    def __init__(self, expl_var, choice_var, split, config=None):
        if not config:
            config = DEFAULT_CONFIG
        
        self.input = T.matrix('x')
        self.output = T.imatrix('y')

        self.n_obs = expl_var.shape[0]
        self.n_vars = expl_var.shape[1]

        train_input = expl_var.iloc[:, :int(split*n_obs)]
        valid_input = expl_var.iloc[:, int(split*n_obs):]

        train_choice = choice_var.iloc[:, :int(split*n_obs)]
        valid_choice = choice_var.iloc[:, int(split*n_obs):]

        self.train_input, self.train_choice = make_shared(train_input, train_choice)
        self.valid_input, self.valid_choice = make_shared(valid_input, valid_choice)