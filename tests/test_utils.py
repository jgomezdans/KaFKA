#!/usr/bin/env python
import datetime
import os
import sys

from distutils import dir_util

import numpy as np

from pytest import fixture

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

from kafka.utils import iterate_time_grid


def test_iterate_time_grid():
    base_date = datetime.datetime(2007, 7, 1)
    time_grid = [base_date + i*datetime.timedelta(days=1) for i in range(
        0, 60, 16)]
    base_date = datetime.datetime(2007, 1, 1)
    the_dates = [base_date + i*datetime.timedelta(days=1) for i in range(
        1, 365+8, 8)]
    print time_grid
    timesteps_good = [datetime.datetime(2007, 7, 17),
                      datetime.datetime(2007, 8, 2),
                      datetime.datetime(2007, 8, 18)]
    obs_times = [np.array([datetime.datetime(2007, 7, 5, 0, 0),
                           datetime.datetime(2007, 7, 13, 0, 0)]),
                 np.array([datetime.datetime(2007, 7, 21, 0, 0),
                           datetime.datetime(2007, 7, 29, 0, 0)]),
                 np.array([datetime.datetime(2007, 8, 6, 0, 0),
                           datetime.datetime(2007, 8, 14, 0, 0)])]

    for i, retval in enumerate(iterate_time_grid(time_grid, the_dates)):
        assert timesteps_good[i] == retval[0]
        assert np.all(obs_times[i] == retval[1])
