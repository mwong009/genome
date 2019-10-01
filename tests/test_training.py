#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd


def test_load_data():
    raw_data = pd.read_csv("data/test_data.csv")
    x_data = raw_data.iloc[:, 1:-1]
    y_data = raw_data['mode']-1  # -1 for indexing at 0

    check_columns = [
        'weekend', 'hour_8_10', 'hour_11_13', 'hour_14_16', 'hour_17_19',
        'hour_20_22', 'hour_23_1', 'hour_2_4', 'hour_5_7', 'num_coord',
        'trip_dist', 'trip_time', 'trip_aspeed', 'act_edu', 'act_health',
        'act_leisure', 'act_meal', 'act_errand', 'act_shop', 'act_home',
        'act_work', 'act_meeting',
    ]
    assert all([a == b for a, b in zip(x_data.columns, check_columns)])
