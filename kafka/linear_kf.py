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

from collections import namedtuple
import numpy as np
import scipy.sparse as sp

from utils import  matrix_squeeze, spsolve2, reconstruct_array
from solvers import linear_diagonal_solver

# Set up logging
import logging
LOG = logging.getLogger(__name__+".linear_kf")


__author__ = "J Gomez-Dans"
__copyright__ = "Copyright 2017 J Gomez-Dans"
__version__ = "1.0 (09.03.2017)"
__license__ = "GPLv3"
__email__ = "j.gomez-dans@ucl.ac.uk"

Metadata = namedtuple('Metadata', 'mask uncertainty')


class LinearKalman (object):
    """The main Kalman filter class operating in raster data sets. Note that the
    goal of this class is not to consider complex, time evolving models, but
    rather grotty "0-th" order models!"""
    def __init__(self, observations, observation_times,
                observation_metadata, output_array, output_unc,
                 n_params=1, diagnostics=True, bands_per_observation=1):
        """The class creator takes a list of observations, some metadata and a
        pointer to an output array."""
        self.n_params = n_params
        self.observations = observations
        self.observation_times = observation_times
        self.metadata = observation_metadata
        self.output = output_array
        self.output_unc = output_unc
        self.diagnostics = diagnostics
        self.bands_per_observation = bands_per_observation

    def _set_plot_view (self, diag_string, timestep, obs):
        """This sets out the plot view for each iteration. Please override this
        method with whatever you want."""
        pass

    def _plotter_iteration_start (self, plot_obj, x, obs, mask):
        """We call this diagnostic method at the **START** of the iteration"""
        pass

    def _plotter_iteration_end (self, plot_obj, x, P, innovation, mask):
        """We call this diagnostic method at the **END** of the iteration"""
        pass

    def _dump_output(self, step, timestep, x_analysis, P_analysis,
                     P_analysis_inverse):
        """Store the output somewhere for further use. This method is called
        after each time step, so if several observations are available within
        the same timestep, it will be the combined result of all observations.
        Currently, only outputting to numpy arrays...
        Parameters
        ----------
        step: integer
            The "band" in the output array
        timestep: integer
            The actual timestep (e.g. DoY or something like that)
        x_analysis: array
            The analysis state vector
        P_analysis: array
            The analysis state covariance
        """
        # Needs to take self.n_params into account
        if self.n_params > 1:
            N = self.output.shape[2] * self.output.shape[3]
            for param in xrange(self.n_params):
                self.output[step, param, :, :] = x_analysis[
                                               (param * N):(
                                               (param + 1) * N)].reshape(
                    (self.output.shape[2:]))
                self.output_unc[step, param, :, :] = P_analysis.diagonal()[
                                                   (param * N):(
                                                   (param + 1) * N)].reshape(
                    (self.output.shape[2:]))
        else:
            self.output[step, :, :] = x_analysis.reshape((self.output.shape[1:]))
            self.output_unc[step, :, :] = P_analysis.diagonal().reshape(
                self.output.shape[1:])

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
        # Start with the mask
        mask = self.metadata[timestep].mask
        # Fish out the observations
        if band is None:
            observations = self.observations[timestep]
            R_mat = self.create_uncertainty(
                self.metadata[timestep].uncertainty, mask)

        else:
            observations = self.observations[timestep][band]
            R_mat = self.create_uncertainty(
                self.metadata[timestep].uncertainty[band], mask)
        return observations, R_mat, mask.ravel(), self.metadata[timestep]

    def set_trajectory_model(self, nx, ny):
        """In a Kalman filter, the state is progated from time `t` to `t+1`
        using a model. We assume that this model is a matrix, and for the time
        being, the matrix is the identity matrix. That's how we roll!"""
        n = nx*ny
        self.trajectory_model = sp.eye(self.n_params*n, self.n_params*n,
                                       format="csr")

    def set_trajectory_uncertainty(self, Q, nx, ny):
        """In a Kalman filter, the model that propagates the state from time
        `t` to `t+1` is assumed to be *wrong*, and this is indicated by having
        additive Gaussian noise, which we assume is zero-mean, and controlled by
        a covariance matrix `Q`. Here, you can provide the main diagonal of `Q`.

        Parameters
        -----------
        Q: array
            The main diagonal of the model uncertainty covariance matrix.
        """
        n = nx*ny 
        self.trajectory_uncertainty = sp.eye(self.n_params*n, self.n_params*n,
                                       format="csr")
        self.trajectory_uncertainty.setdiag(Q)

    def create_uncertainty(self, uncertainty, mask):
        """Creates the observational uncertainty matrix. We assume that
        uncertainty is a single value and we return a diagonal matrix back.
        We present this diagonal **ONLY** for pixels that have observations
        (i.e. not masked)."""
        good_obs = mask.sum()
        R_mat = np.ones (good_obs)*uncertainty*uncertainty
        return sp.dia_matrix((R_mat, 0), shape=(R_mat.shape[0], R_mat.shape[0]))

    def create_observation_operator (self, metadata, x_forecast, band=None):
        """A simple **identity** observation opeartor. It is expected that you
        subclass and redefine things...."""
        good_obs = metadata.mask.sum() # size of H_matrix
        H_matrix = sp.dia_matrix(np.eye (good_obs))
        return H_matrix

    def advance(self, x_analysis, P_analysis, P_analysis_inverse):
        """Advance the state"""
        # Needs to deal with the inverse analysis matrix ;-/
        # If P_analysis_inverse is block_diagonal.... EASY
        # see notes
        x_forecast = self.trajectory_model.dot(x_analysis)
        if sp.issparse(self.trajectory_uncertainty) and P_analysis is not None:
            P_forecast = P_analysis + self.trajectory_uncertainty
            P_forecast_inverse = None
        elif sp.issparse(self.trajectory_uncertainty) and P_analysis is None:
            LOG.info("Updating prior *invese covariance*")
            LOG.info("Only updates main diagonal")
            diag = 1./P_analysis_inverse.diagonal()
            P_approx = sp.dia_matrix((diag, 0), 
                                     shape=P_analysis_inverse.shape) + \
                self.trajectory_uncertainty
            P_forecast_inverse = P_analysis_inverse.copy()
            P_forecast_inverse.setdiag( 1./P_approx.diagonal() )
            P_forecast = None
            
        else:
            trajectory_uncertainty = sp.dia_matrix((self.trajectory_uncertainty,
                                                    0), shape=P_analysis.shape)
            P_forecast = P_analysis + trajectory_uncertainty
            P_forecast_inverse = None

        return x_forecast, P_forecast, P_forecast_inverse

    def run(self, x_forecast, P_forecast, P_forecast_inverse,
                   diag_str="diagnostics",
                   band=None, approx_diagonal=True, refine_diag=True,
                   iter_obs_op=False, is_robust=False):
        is_first = True
        for ii,timestep in enumerate(np.arange(self.observation_times.min(),
                                  self.observation_times.max() + 1)):
            # First locate all available observations for time step of interest.
            # Note that there could be more than one...
            locate_times = [i for i, x in enumerate(self.observation_times)
                        if x == timestep]
            
            LOG.info("timestep %d" % timestep)
            
            if not is_first:
                LOG.info("Advancing state, timestep %d" % timestep)
                x_forecast, P_forecast, P_forecast_inverse = self.advance(
                    x_analysis, P_analysis, P_analysis_inverse)
            is_first = False
            if len(locate_times) == 0:
                # Just advance the time
                x_analysis = x_forecast
                P_analysis = P_forecast
                P_analysis_inverse = P_forecast_inverse
                LOG.info("No observations in this time")
                
            else:
                # We do have data, so we assimilate
                LOG.info("# of Observations: %d" % len(locate_times))

                x_analysis, P_analysis, P_analysis_inverse = self.assimilate (
                                     locate_times, x_forecast, P_forecast,
                                     P_forecast_inverse,
                                     approx_diagonal=approx_diagonal,
                                     refine_diag=refine_diag,
                                     iter_obs_op=iter_obs_op,
                                     is_robust=is_robust, diag_str=diag_str)

            self._dump_output(ii, timestep, x_analysis, P_analysis, 
                              P_analysis_inverse)

    def assimilate(self, locate_times, x_forecast, P_forecast, 
                   P_forecast_inverse,
                   approx_diagonal=True, refine_diag=False,
                   iter_obs_op=False, is_robust=False, diag_str="diag"):
        """The method assimilates the observatins at timestep `timestep`, using
        a prior a multivariate Gaussian distribution with mean `x_forecast` and
        variance `P_forecast`."""
        for step in locate_times:
            LOG.info("Assimilating %d..." % step)
            # This first loop iterates the solution for all bands
            # We store the forecast to compare convergence after one 
            # iteration
            x_prev = x_forecast*1.
            converged = False
            n_iter = 0
            while True:
                if n_iter == 0:
                    # Read in the data for all bands so we don't have 
                    # to read it many times.
                    cached_obs = []
                    for band in xrange(self.bands_per_observation):
                        if self.bands_per_observation == 1:
                            observations, R_mat, mask, the_metadata = \
                                self._get_observations_timestep(step, None)
                        else:
                            observations, R_mat, mask, the_metadata = \
                                self._get_observations_timestep(step, band)
                        cached_obs.append ( (observations, R_mat, mask, 
                                           the_metadata) )
                        
                n_iter += 1
                for band in xrange(self.bands_per_observation):
                    LOG.info("Band %d" % band)
                    # Extract observations, mask and uncertainty for the current time
                    #if self.bands_per_observation == 1:
                        #observations, R_mat, mask, the_metadata = \
                            #self._get_observations_timestep(step, None)
                    #else:
                        #observations, R_mat, mask, the_metadata = \
                            #self._get_observations_timestep(step, band)
                    observations, R_mat, mask, the_metadata = cached_obs[band]
                    if self.diagnostics:
                        LOG.info("Setting up diagnostics...")
                        plot_object = self._set_plot_view(diag_str, step, observations)
                        self._plotter_iteration_start(plot_object, x_forecast,
                                                      observations, mask )
                    if self.bands_per_observation == 1:
                        # Remember that x_prev is the value that the iteration
                        # is working on. Starts with x_forecast, but updated
                        H_matrix = self.create_observation_operator(the_metadata,
                                                    x_prev, None)
                    else:
                        H_matrix = self.create_observation_operator(the_metadata,
                                                    x_prev, band)
                    # the mother of all function calls
                    x_analysis, P_analysis, P_analysis_inverse, \
                        innovations_prime  = self.solver(
                            observations, mask, H_matrix,
                            x_forecast, P_forecast, P_forecast_inverse, 
                            R_mat, the_metadata)

                    x_forecast = x_analysis*1
                    P_forecast = P_analysis
                    P_forecast_inverse = P_analysis_inverse                    





                if iter_obs_op:
                    # this should be an option...
                    maska = np.concatenate([mask.ravel() 
                                            for i in xrange(self.n_params)]) 
                    convergence_norm = np.linalg.norm(x_analysis[maska] - 
                                            x_prev[maska])/float(maska.sum())
                    if convergence_norm <= 5e-4:
                        converged = True
                        LOG.info("Converged (%g) !!!"%convergence_norm)
                    x_prev = x_analysis*1.
                    LOG.info("Iteration %d convergence: %g" %( n_iter, 
                                                            convergence_norm))
        #                if is_robust and converged:
        #                    break
        #                    # TODO robust re-masking
        #                    # We should have a robust mechanism that checks whether the state
        #                    # is too far from the observations, and if so, flag them as
        #                    # outliers
                else:
                    break

                if converged and n_iter > 2:
                    # Convergence, getting out of loop
                    # Store current state as x_forecast in case more obs today
                    # the analysis becomes the forecast for the next
                    # iteration
                    
                    break

                if n_iter >= 20:
                    # Break if we go over 10 iterations
                    LOG.info("Wow, too many iterations (%d)!"%n_iter)
                    LOG.info("Stopping iterations here")
                    break
        if self.diagnostics:
            LOG.info("Plotting")
            self._plotter_iteration_end(plot_object, x_analysis,
                                        P_analysis, innovations_prime, mask)


        return x_analysis, P_analysis, P_analysis_inverse

    def solver(self, observations, mask, H_matrix, x_forecast, P_forecast,
                P_forecast_inverse, R_mat, the_metadata):

        x_analysis, P_analysis, P_analysis_inverse, innovations_prime = \
            linear_diagonal_solver (
                observations, mask, H_matrix, self.n_params, x_forecast,
                P_forecast, R_mat, the_metadata)

        return x_analysis, P_analysis, P_analysis_inverse, innovations_prime


