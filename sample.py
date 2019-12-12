import pandas as pd
import theano
import theano.tensor as T

from genome import Genome
from genome.models.logit import MultinomialLogit

# read and prepare data
df = pd.read_csv('data/test_data')
dataset = df[['weekend', 'hour_8_10', 'hour_11_13', 'trip_dist', 'mode']]

# readjusting index for output choice
# choice index needs to start from 0
dataset[['mode']] = dataset[['mode']] - 1

# set up Genome with dataset and configuration
gn = Genome(data=dataset, choice='mode', split=0.7, config=None)

gn.choices = {
    0: 'auto',
    1: 'bike',
    2: 'transit',
    3: 'walk',
    4: 'auto+transit',
    5: 'other_mode',
    6: 'other_combination',
}

# define the model
model = MultinomialLogit(input=gn.input, n_vars=gn.n_vars, n_choices=gn.choices)

# build the model
gn.build(model)
gn.set_hyperparameter('validation_freq', gn.config['n_train_batches'])

# run the model training algorithm
gn.run(model)