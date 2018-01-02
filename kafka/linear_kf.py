#!/usr/bin/env python
"""A fast Kalman filter implementation designed with raster data in mind. This
implementation basically performs a very fast update of the filter."""

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

import logging
from collections import namedtuple

import numpy as np

import scipy.sparse as sp

# from scipy.spatial.distance import squareform, pdist

# from utils import  matrix_squeeze, spsolve2, reconstruct_array
from solvers import variational_kalman
from utils import locate_in_lut, run_emulator, create_uncertainty
from utils import create_linear_observation_operator
from utils import create_nonlinear_observation_operator
from utils import iterate_time_grid
from kf_tools import hessian_correction, propagate_information_filter_SLOW

# Set up logging

LOG = logging.getLogger(__name__+".linear_kf")


__author__ = "J Gomez-Dans"
__copyright__ = "Copyright 2017 J Gomez-Dans"
__version__ = "1.0 (09.03.2017)"
__license__ = "GPLv3"
__email__ = "j.gomez-dans@ucl.ac.uk"

Metadata = namedtuple('Metadata', 'mask uncertainty')
Previous_State = namedtuple("Previous_State",
                            "timestamp x_vect cov_m icov_mv")


class LinearKalman (object):
    """The main Kalman filter class operating in raster data sets. Note that the
    goal of this class is not to consider complex, time evolving models, but
    rather grotty "0-th" order models!"""
    def __init__(self, observations, output, state_mask,
                 state_propagation=propagate_information_filter_SLOW,
                 linear=True, n_params=1, diagnostics=True,
                 bands_per_observation=1):
        """The class creator takes (i) an observations object, (ii) an output
        writer object, (iii) the state mask (a boolean 2D array indicating which
        pixels are used in the inference), and additionally, (iv) a state
        propagation scheme (defaults to `propagate_information_filter`),
        whether a linear model is used or not, the number of parameters in
        the state vector, whether diagnostics are being reported, and the
        number of bands per observation.
        """
        self.n_params = n_params
        self.observations = observations
        self.output = output
        self.diagnostics = diagnostics
        self.bands_per_observation = bands_per_observation
        self.state_mask = state_mask
        self.n_state_elems = self.state_mask.sum()
        self._advance = state_propagation
        if linear:
            self._create_observation_operator = \
                                            create_linear_observation_operator
        else:
            self._create_observation_operator = \
                                        create_nonlinear_observation_operator
        LOG.info("Starting KaFKA run!!!")

    def advance(self, x_analysis, P_analysis, P_analysis_inverse,
                    trajectory_model, trajectory_uncertainty):
        LOG.info("Calling state propagator...")
        x_forecast, P_forecast, P_forecast_inverse = \
            self._advance(x_analysis, P_analysis, P_analysis_inverse,
                          trajectory_model, trajectory_uncertainty)
        return x_forecast, P_forecast, P_forecast_inverse

    def _set_plot_view(self, diag_string, timestep, obs):
        """This sets out the plot view for each iteration. Please override this
        method with whatever you want."""
        pass

    def _plotter_iteration_start(self, plot_obj, x, obs, mask):
        """We call this diagnostic method at the **START** of the iteration"""
        pass

    def _plotter_iteration_end(self, plot_obj, x, P, innovation, mask):
        """We call this diagnostic method at the **END** of the iteration"""
        pass

    def set_trajectory_model(self):
        """In a Kalman filter, the state is progated from time `t` to `t+1`
        using a model. We assume that this model is a matrix, and for the time
        being, the matrix is the identity matrix. That's how we roll!"""
        n = self.n_state_elems
        self.trajectory_model = sp.eye(self.n_params*n, self.n_params*n,
                                       format="csr")

    def set_trajectory_uncertainty(self, Q):
        """In a Kalman filter, the model that propagates the state from time
        `t` to `t+1` is assumed to be *wrong*, and this is indicated by having
        additive Gaussian noise, which we assume is zero-mean, and controlled
        by a covariance matrix `Q`. Here, you can provide the main diagonal of
         `Q`.

        Parameters
        -----------
        Q: array
            The main diagonal of the model uncertainty covariance matrix.
        """
        n = self.n_state_elems
        self.trajectory_uncertainty = sp.eye(self.n_params*n, self.n_params*n,
                                             format="csr")
        self.trajectory_uncertainty.setdiag(Q)

    def _get_observations_timestep(self, timestep, band=None):
        """A method that returns the observations, mask and uncertainty for a
        particular timestep. It is envisaged that applications will specialise
        this method to efficiently read and process raster datasets from disk,
        but this version will just select from a numpy array or something.

        Parameters
        ----------
        timestep: int
            This is the time step that we require the information for.
        band: int
            For multiband datasets, this selects the band to use, or `None` if
            single band dataset is used.

        Returns
        -------
        Observations (N*N), uncertainty (N*N) and mask (N*N) arrays, as well
        as relevant metadata
        """
        data = self.observations.get_band_data(timestep, band)
        return (data.observations, data.uncertainty, data.mask,
                data.metadata, data.emulator)

    def run(self, time_grid, x_forecast, P_forecast, P_forecast_inverse,
            diag_str="diagnostics",
            band=None, approx_diagonal=True, refine_diag=True,
            iter_obs_op=False, is_robust=False, dates=None):
        """Runs a complete assimilation run. Requires a temporal grid (where
        we store the timesteps where the inferences will be done, and starting
        values for the state and covariance (or inverse covariance) matrices.

        The time_grid ought to be a list with the time steps given in the same
        form as self.observation_times"""
        for timestep, locate_times, is_first in iterate_time_grid(
            time_grid, self.observations.dates):

            self.current_timestep = timestep

            if not is_first:
                LOG.info("Advancing state, %s" % timestep.strftime("%Y-%m-%d"))
                x_forecast, P_forecast, P_forecast_inverse = self.advance(
                    x_analysis, P_analysis, P_analysis_inverse,
                    self.trajectory_model, self.trajectory_uncertainty)
            is_first = False
            if len(locate_times) == 0:
                # Just advance the time
                x_analysis = x_forecast
                P_analysis = P_forecast
                P_analysis_inverse = P_forecast_inverse
                LOG.info("No observations in this time")

            else:
                # We do have data, so we assimilate

                x_analysis, P_analysis, P_analysis_inverse = self.assimilate(
                                     locate_times, x_forecast, P_forecast,
                                     P_forecast_inverse,
                                     approx_diagonal=approx_diagonal,
                                     refine_diag=refine_diag,
                                     iter_obs_op=iter_obs_op,
                                     is_robust=is_robust, diag_str=diag_str)
            LOG.info("Dumping results to disk")
            self.output.dump_data(timestep, x_analysis, P_analysis,
                                  P_analysis_inverse, self.state_mask)

    def assimilate(self, locate_times, x_forecast, P_forecast,
                   P_forecast_inverse,
                   approx_diagonal=True, refine_diag=False,
                   iter_obs_op=False, is_robust=False, diag_str="diag"):
        """The method assimilates the observatins at timestep `timestep`, using
        a prior a multivariate Gaussian distribution with mean `x_forecast` and
        variance `P_forecast`."""
        for step in locate_times:
            LOG.info("Assimilating %s..." % step.strftime("%Y-%m-%d"))
            # This first loop iterates the solution for all bands
            # We store the forecast to compare convergence after one
            # iteration
            x_prev = x_forecast*1.
            converged = False
            n_iter = 0
            have_obs = True
            while True:
                if n_iter == 0:
                    # Read in the data for all bands so we don't have
                    # to read it many times.
                    cached_obs = []
                    for band in xrange(self.bands_per_observation):
                        LOG.info("\tReading observations in....")
                        if self.bands_per_observation == 1:
                            observations, R_mat, mask, the_metadata, emulator \
                                = self._get_observations_timestep(step, None)
                        else:
                            observations, R_mat, mask, the_metadata, emulator \
                                = self._get_observations_timestep(step, band)
                        cached_obs.append((observations, R_mat, mask,
                                           the_metadata, emulator))
                    if mask.sum() == 0:
                        # No observations!!
                        LOG.info("No observations for band %d" % (band+1))
                        have_obs = False
                        x_analysis = x_forecast*1
                        P_analysis = P_forecast
                        P_analysis_inverse = P_forecast_inverse
                        break
                n_iter += 1

                for band in xrange(self.bands_per_observation):
                    LOG.info("Band %d" % band)
                    # Extract obs, mask and uncertainty for current time
                    # From cache
                    observations, R_mat, mask, the_metadata, the_emulator = \
                        cached_obs[band]

                    if self.diagnostics:
                        LOG.info("Setting up diagnostics...")
                        plot_object = self._set_plot_view(diag_str, step,
                                                          observations)
                        self._plotter_iteration_start(plot_object, x_forecast,
                                                      observations, mask)
                    if self.bands_per_observation == 1:
                        # Remember that x_prev is the value that the iteration
                        # is working on. Starts with x_forecast, but updated
                        H_matrix = self._create_observation_operator(
                            self.n_params, the_emulator, the_metadata,
                            mask, self.state_mask, x_prev, None)
                    else:
                        H_matrix = self._create_observation_operator(
                            self.n_params, the_emulator, the_metadata,
                            mask, self.state_mask, x_prev, band)
                    # the mother of all function calls
                    x_analysis, P_analysis, P_analysis_inverse, \
                        innovations_prime = self.solver(
                            observations, mask, H_matrix,
                            x_forecast, P_forecast, P_forecast_inverse,
                            R_mat, the_metadata)

                    
                    P_correction = hessian_correction(
                                the_emulator, x_analysis, R_mat, 
                                innovations_prime, mask, self.state_mask, band, self.n_params)

                    P_analysis_inverse = P_analysis_inverse - P_correction
                    
                    x_forecast = x_analysis*1
                    P_forecast = P_analysis
                    P_forecast_inverse = P_analysis_inverse*1

                if iter_obs_op:
                    # this should be an option...
                    M = mask[self.state_mask]
                    maska = np.concatenate([M.ravel()
                                            for i in xrange(self.n_params)])
                    convergence_norm = np.linalg.norm(x_analysis[maska] -
                                                      x_prev[maska])/float(
                                                          maska.sum())
                    if convergence_norm <= 1e-3:
                        converged = True
                        LOG.info("Converged (%g) !!!" % convergence_norm)
                    x_prev = x_analysis*1.
                    LOG.info("Iteration %d convergence: %g" % (
                        n_iter, convergence_norm))
                else:
                    break

                if converged and n_iter > 1:
                    # Convergence, getting out of loop
                    # Store current state as x_forecast in case more obs today
                    # the analysis becomes the forecast for the next
                    # iteration
                    break

                if n_iter >= 15:
                    # Break if we go over 10 iterations
                    LOG.info("Wow, too many iterations (%d)!" % n_iter)
                    LOG.info("Stopping iterations here")
                    converged = True
                    break
            if is_robust and converged:
                # TODO update mask using innovations
                pass

            if self.diagnostics and have_obs:
                LOG.info("Plotting")
                self._plotter_iteration_end(plot_object,
                                            x_analysis, P_analysis,
                                            innovations_prime, mask)

        # Store the current state as previous state
        # Rationale is that when called on demand, we need to be able to
        # propagate state to next available observation
        self.previous_state = Previous_State(step, x_analysis,
                                             P_analysis, P_analysis_inverse)

        return x_analysis, P_analysis, P_analysis_inverse

    def solver(self, observations, mask, H_matrix, x_forecast, P_forecast,
               P_forecast_inv, R_mat, the_metadata):

        x_analysis, P_analysis, P_analysis_inv, innovations_prime = \
            variational_kalman(
                observations, mask, self.state_mask, R_mat, H_matrix,
                self.n_params,
                x_forecast, P_forecast, P_forecast_inv, the_metadata)

        return x_analysis, P_analysis, P_analysis_inv, innovations_prime
