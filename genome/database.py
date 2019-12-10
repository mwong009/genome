import numpy as np
import theano

FLOATX = theano.config.floatX

def make_shared(data_x, data_y, borrow=True):

    shared_x = theano.shared(np.asarray(data_x, dtype=FLOATX), borrow=True)
    shared_y = theano.shared(np.asarray(data_y, dtype=FLOATX), borrow=True)
    shared_y = theano.tensor.cast(shared_y, 'int32')

    return shared_x, shared_y