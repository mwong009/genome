#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import genome

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import theano
import theano.tensor as T
import pytest


class TestIntegration():
    @pytest.mark.parametrize('filepath', ['data/test_data.csv'])
    def test_loadData(self, filepath):
        datafile = Path(filepath)
        assert datafile.exists()

    def test_invoke(self):
        sys.argv = ["data/test_data.csv"]

    def test_update(self):
        pass
