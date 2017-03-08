#!/usr/bin/env python
"""Some utility functions to plot diagnostics."""

# KaFKA A fast Kalman filter implementation for raster based datasets.
# Copyright (c) 2017 J Gomez-Dans. All rights reserved.
#
# This file is part of KaFKA.
#
# KaFKA is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# KaFKA is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with KaFKA.  If not, see <http://www.gnu.org/licenses/>.

__author__ = "J Gomez-Dans"
__copyright__ = "Copyright 2017 J Gomez-Dans"
__version__ = "1.0 (09.03.2017)"
__license__ = "GPLv3"
__email__ = "j.gomez-dans@ucl.ac.uk"


import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import os

def distance_kullback(A, B):
    """Kullback Leibler divergence between two covariance matrices A and B.
    Formula here:
    https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

    Parameters
    ----------
    A : One covariance matrix
    B : The other

    Returns
    -------
    The KL divergence ;) in nats. Divide by log_e 2 to get bits
    """

    dim = A.shape[0]
    logdet = np.log(np.linalg.det(B) / np.linalg.det(A))
    kl = np.trace(np.dot(np.linalg.inv(B), A)) - dim + logdet
    return 0.5 * kl

def plot_observations (fstring, step, observations, mask):
    """
    A simple plotting routine to plot the observations at every assimilation
    stage.
    Parameters
    ----------
    fstring : The simulation name string
    step : The time step of the assimilation
    observations : The observations
    mask : The mask

    Returns
    -------
    Nuffinkg
    """
    cmap = plt.cm.viridis
    ny, nx = observations.shape
    mask = mask.reshape((ny,nx))
    fig, axs = plt.subplots(nrows=1, ncols=1)
    axs = axs.flatten()
    axs[0].imshow ( observations, interpolation='nearest', cmap=cmap)
    axs[1].imshow(mask, interpolation='nearest',cmap=plt.cm.gray)
    fig.savefig("obs_%s_%05d.png" % (fstring, step), dpi=150, bbox_inches="tight")
    fig.close()

def plot_innovations(fstring, step, innovation, mask, obs_uncertainty):
    """
    Plots the innovations
    Parameters
    ----------
    fstring :
    step :
    innovation :
    mask :
    obs_uncertainty :

    Returns
    -------

    """

def plot_posterior_prior_update():
    # Basically KL divergence
    # Can probably be done per pixel and shown as an image
    pass