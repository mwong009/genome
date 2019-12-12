"""Genome"""
from genome.base import Environment
from genome.events import Setup
from genome.optimizers import *
from genome.logging import *

from genome.database import shared

import theano
import theano.tensor as T
import timeit

DEFAULT_CONFIG = {
    'batch_size': 32,
    'n_epochs': 250,
    'patience': 50000,
    'patience_inc': 2,
    'patience_threshold': 0.995,
    'learning_rate': 0.01,
    'decay': 0.05,
    'validation_freq': 100,
}

DEFAULT_OPTIMIZERS = {
    'SGD': SGD,
    'MomentumSGD': MomentumSGD,
    'Adagrad': Adagrad,
    'RMSProp': RMSProp,
    'Adadelta': Adadelta,
    'Adam': Adam,
}

class Genome(object):
    def __init__(self, data, choice, split, config=None):
        if not config:
            config = DEFAULT_CONFIG
        
        self.config = config

        # define the symbolic input variable
        self.input = T.matrix('x')

        # define the symbolic output variable
        self.output = T.imatrix('y')

        choice_var = data[[choice]]
        expl_var = data.loc[:, data.columns != choice]
        
        self.n_obs = expl_var.shape[0]
        self.n_vars = expl_var.shape[1]

        train_input = expl_var.iloc[:, :int(split*n_obs)]
        valid_input = expl_var.iloc[:, int(split*n_obs):]

        self.config['n_train_obs'] = train_input.shape[0]
        self.config['n_valid_obs'] = valid_input.shape[0]

        n_train_batches = self.config['n_train_obs'] // self.config['batch_size']
        self.config['n_train_batches'] = n_train_batches

        train_choice = choice_var.iloc[:, :int(split*n_obs)]
        valid_choice = choice_var.iloc[:, int(split*n_obs):]

        self.train_input, self.train_choice = shared(train_input, train_choice)
        self.valid_input, self.valid_choice = shared(valid_input, valid_choice)
    
    def set_hyperparameter(self, name, value):
        if name in self.config:
            self.config[name] = value
        else:
            raise KeyError('Warning! ' + name + ' is not a valid hyperparameter value.')

    def build(self, model, optimizer='SGD'):
        i = T.iscalar('minibatch_index')
        batch_size = self.config['batch_size']
        
        if optimizer in DEFAULT_OPTIMIZERS:
            opt = DEFAULT_OPTIMIZERS[optimizer]
            model.opt = opt(model.params)
        else:
            raise KeyError('Optimizer \'' + optimizer + '\' not found!')

        cost = model.negative_log_likelihood(self.output)
        updates = model.opt.update(model.cost, model.params)

        print('constructing the computational graph...')
        self.train_fn = theano.function(
            input=[i],
            output=cost,
            updates=updates,
            givens= {
                self.input: self.train_input[i*batch_size: (i+1)*batch_size],
                self.output: self.train_choice[i*batch_size: (i+1)*batch_size],
            },
            allow_input_downcast=True,
            on_unused_input='ignore',
            name='train_fn',
        )

        self.valid_fn = theano.function(
            inputs=[],
            outputs=cost,
            updates=updates,
            givens={
                self.input: self.valid_input,
                self.output: self.valid_choice,
            },
            allow_input_downcast=True,
            on_unused_input='ignore',
            name='valid_fn',
        )
        print('...build complete.')

    def run(self, model, **kwargs):
        config = self.config
        
        done_looping = False
        logging = None
        epoch = 0
        self.best_train_score = np.inf
        self.best_valid_score = np.inf
        patience = config['patience']
        threshold = config['patience_threshold']
        patience_inc = config['patience_inc']
        
        if 'epoch' in kwargs:
            epoch = kwargs['epoch']

        start_time = timeit.default_timer()

        while (epoch < config['n_epochs']) and (not done_looping):

            train_score = []
            for n in config['n_train_batches']:
                score = self.train_fn(n)
                train_score.append(score)

                iterations = epoch * config['n_train_batches'] + n

                if (iterations + 1) % config['validation_freq'] == 0:
                    valid_score = self.valid_fn()

                    if valid_score < self.best_valid_score * threshold:
                        patience = max(patience, iterations * patience_inc)
                        
                    if valid_score < self.best_valid_score:
                        self.best_valid_score = valid_score
                        self.best_train_score = np.mean(train_score)

                    logging = {
                        'epoch': epoch,
                        'iterations': n,
                        'learning_rate': config['learning_rate'],
                        'train_score': np.mean(train_score),
                        'valid_score': valid_score,
                        'training_time': (timeit.default_timer()-start_time)/60.,
                    }
                
                if (iterations + 1) % config['n_train_batches'] == 0:
                    if logging is not None:
                        print_score(logging)

                if (patience <= iterations) & (epoch > 10):
                    done_looping = True
                    break
            
            epoch = epoch + 1
        
        end_time = timeit.default_timer()

        print('training completed.')