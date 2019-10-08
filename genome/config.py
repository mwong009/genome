#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genome Config Module

This module is used for loading and saving of configuration files, reading and writing
of datafiles.

Reading datafiles
-----------------
Some text.
"""
import pandas as pd


def load_data(filename):
    """Reads data from `filename`

    Parameters
    ----------
    filename: int
        input .csv file to be read
    """
    data = pd.read_csv(filename)
    return data


