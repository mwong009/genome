import pandas as pd
import theano
import theano.tensor as T

from genome import Genome
from genome.models.logit import MultinomialLogit

df = pd.read_csv('data/test_data')

x_data = df[['weekend', 'hour_8_10', 'hour_11_13', 'trip_dist']]
y_data = df[['mode']] - 1

# set up Genome with config dictionary file
gn = Genome(expl_var=x_data, choice_var=y_data, split=0.7, config=None)

gn.choices = {
    0: 'auto',
    1: 'bike',
    2: 'transit',
    3: 'walk',
    4: 'auto+transit',
    5: 'other_mode',
    6: 'other_combination',
}

model = MultinomialLogit(gn.input, gn.n_vars, gn.choices)

gn.run()