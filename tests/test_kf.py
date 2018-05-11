#!/usr/bin/env python
import datetime
import os
import sys

from distutils import dir_util

import numpy as np

from pytest import fixture

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

from kafka.inference.kf_tools import propagate_standard_kalman
from kafka.inference.kf_tools import propagate_information_filter


def test_propagate_standard_kalman():
    x_analysis = np.ones(3)
    P_analysis = np.eye(3)
    M_matrix = 2.*np.eye(3)
    Q_matrix = np.eye(3)*0.5
    x_forecast, P_forecast, P_forecast_inverse = propagate_standard_kalman(
         x_analysis, P_analysis, None, M_matrix, Q_matrix)
    assert np.all(x_forecast == 2.*x_analysis)
    assert np.all(P_forecast == (P_analysis + np.eye(3)*0.5))


def test_propagate_information_filter():
    np.set_printoptions(precision=2, linewidth=132)

    M_matrix = np.eye(7)
    sigma = np.array([0.12, 0.7, 0.0959, 0.15, 1.5, 0.2, 0.5])
    x_analysis = np.array([0.17, 1.0, 0.1, 0.7, 2.0, 0.18, np.exp(-0.5*1.5)])
    Pd = np.diag(sigma**2).astype(np.float32)
    Pd[5, 2] = 0.8862*0.0959*0.2
    Pd[2, 5] = 0.8862*0.0959*0.2
    Pi = np.linalg.inv(Pd)
    Q_matrix = np.eye(7)*0.1

    x_forecast, P_forecast, P_forecast_inverse = propagate_information_filter(
        x_analysis, None, Pi, M_matrix, Q_matrix)
    assert np.allclose(
        np.array(P_forecast_inverse.todense()).squeeze().diagonal(),
        np.array([8.74, 1.69, 9.81, 8.16, 0.43, 9.21, 2.86]), atol=0.01)
    # In reality, the matrix ought to be
    # [[ 8.74  0.    0.    0.    0.    0.    0.  ]
    # [ 0.    1.69  0.    0.    0.    0.    0.  ]
    # [ 0.    0.    9.33  0.    0.   -1.13  0.  ]
    # [ 0.    0.    0.    8.16  0.    0.    0.  ]
    # [ 0.    0.    0.    0.    0.43  0.    0.  ]
    # [ 0.    0.   -1.13  0.    0.    7.28  0.  ]
    # [ 0.    0.    0.    0.    0.    0.    2.86]]
