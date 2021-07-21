import os
import random
import pytest
import genome.dataset as ds


@pytest.fixture
def working_dir():
    return os.getcwd()

@pytest.fixture
def dataset(working_dir):
    data_dir = os.path.join(working_dir, "data")
    data_filename = os.path.join(data_dir, "swissmetro.dat")
    data_sep = "\t"
    return ds.Dataset("dataset", data_filename, data_sep)

@pytest.fixture
def ex_column(dataset):
    c = dataset.pandasData.columns
    return random.sample(list(c), 3)

def test_loaddataset(dataset):
    assert len(dataset.pandasData) == 10728

def test_exclude_columns(ex_column, dataset, working_dir):
    # return
    df = dataset.exclude_columns(*ex_column)
    for column in ex_column:
        assert column not in df.columns
    
    # inplace
    data_dir = os.path.join(working_dir, "data")
    data_filename = os.path.join(data_dir, "swissmetro.dat")
    data_sep = "\t"
    d = ds.Dataset("dataset", data_filename, data_sep)
    d.exclude_columns(*ex_column, inplace=True)
    for column in ex_column:
        assert column not in d.pandasData.columns