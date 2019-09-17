import pytest
import theano
import theano.tensor as T
import genome


def test_import():
    print(theano.config)
    x = T.scalar('x')
    y = T.matrix('y')
    assert 0


def test_joke():
    assert genome.joke('melvin') == 'this is the name: melvin'


def test_run():
    run_object = genome.Run(34)
    assert run_object.compile() == 34
