"""
Extracted the state propagation bits to individual functions
"""
import numpy as np

import scipy.sparse as sp


def propagate_standard_kalman(x_analysis, P_analysis, P_analysis_inverse,
                              M_matrix, Q_matrix):
    x_forecast = M_matrix.dot(x_analysis)
    P_forecast = P_analysis + Q_matrix
    return x_forecast, P_forecast, None


def propagate_information_filter(x_analysis, P_analysis, P_analysis_inverse,
                                 M_matrix, Q_matrix):
    x_forecast = M_matrix.dot(x_analysis)
    # These is an approximation to the information filter equations
    # (see e.g. Terejanu's notes)
    M = P_analysis_inverse   # for convenience and to stay with
    #  Terejanu's notation
    # Main assumption here is that the "inflation" factor is
    # calculated using the main diagonal of M
    D = 1./(1. + M.diagonal()*Q_matrix.diagonal())
    M = sp.dia_matrix((M.diagonal(), 0), shape=M.shape)
    P_forecast_inverse = M.dot(sp.dia_matrix((D, 0),
                                             shape=M.shape))
    return x_forecast, None, P_forecast_inverse
