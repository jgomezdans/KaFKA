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

from .inference import variational_kalman
from .inference import variational_kalman_multiband
from .inference import locate_in_lut, run_emulator, create_uncertainty
from .inference import create_linear_observation_operator
from .inference import create_nonlinear_observation_operator
from .inference import iterate_time_grid
from .inference import hessian_correction
from .inference import hessian_correction_multiband
from .inference import robust_inflation

from .state_propagation import forward_state_propagation



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
                 create_observation_operator, parameters_list,
                 state_propagation,
                 band_mapper=None,
                 linear=True, diagnostics=True, prior=None,
                 upper_bound=None, lower_bound=None):
        """The class creator takes (i) an observations object, (ii) an output
        writer object, (iii) the state mask (a boolean 2D array indicating which
        pixels are used in the inference), and additionally, (iv) a state
        propagation scheme (defaults to `propagate_information_filter`),
        whether a linear model is used or not, the number of parameters in
        the state vector, whether diagnostics are being reported, and the
        number of bands per observation.
        """
        self.parameters_list = parameters_list # A list of parameter names
                                     # Required by prior
        self.n_params = len(self.parameters_list)
    
        self.observations = observations
        self.output = output
        self.diagnostics = diagnostics
        self.state_mask = state_mask
        self.n_state_elems = self.state_mask.sum()
        self._state_propagator = state_propagation
        self._advance = forward_state_propagation
        self.prior = prior
        # The band mapper allows some parameters to be defined per band,
        # such as optical properties per band and so on. The format for
        # this is a list, where each element is an array with the state
        # elements that belong to that particular band (the band being
        # the position in the list).
        self.band_mapper = band_mapper
        # e.g band_mapper for JRC-TIP: [np.array([0, 1, 6, 2]),
        #                               np.array([3, 4, 6, 5])]
        
        # this allows you to pass additional information with prior needed by
        # specific functions. All priors need a dictionary with ['function'] key.
        # Other keys are optional
        self._create_observation_operator = create_observation_operator
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        LOG.info("Starting KaFKA run!!!")

    def advance(self, x_analysis, P_analysis, P_analysis_inverse):
        LOG.info("Polling propagation model")
        trajectory_model, trajectory_uncertainty = self._state_propagator(
            x_analysis, P_analysis, P_analysis_inverse,
            self.current_timestep)
        
        LOG.info("Calling state propagator...")
        
        x_forecast, P_forecast, P_forecast_inverse = \
            self._advance(x_analysis, P_analysis, P_analysis_inverse,
                          trajectory_model, trajectory_uncertainty,
                          self.current_timestep, self.prior)
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

                x_analysis, P_analysis, P_analysis_inverse = self.assimilate_multiple_bands(
                                     locate_times, x_forecast, P_forecast,
                                     P_forecast_inverse,
                                     approx_diagonal=approx_diagonal,
                                     refine_diag=refine_diag,
                                     iter_obs_op=iter_obs_op,
                                     is_robust=is_robust, diag_str=diag_str)
            LOG.info("Dumping results to disk")
            self.output.dump_data(timestep, x_analysis, P_analysis,
                                  P_analysis_inverse, self.state_mask, self.n_params)

    def assimilate_multiple_bands(self, locate_times, x_forecast, P_forecast,
                   P_forecast_inverse,
                   approx_diagonal=True, refine_diag=False,
                   iter_obs_op=False, is_robust=False, diag_str="diag"):        
        """The method assimilates the observatins at timestep `timestep`, using
        a prior a multivariate Gaussian distribution with mean `x_forecast` and
        variance `P_forecast`. THIS DOES ALL BANDS SIMULTANEOUSLY!!!!!"""
        for step in locate_times:
            print("Assimilating %s..." % step.strftime("%Y-%m-%d"))
            LOG.info("Assimilating %s..." % step.strftime("%Y-%m-%d"))
            current_data = []
            # Reads all bands into one list
            for band in range(self.observations.bands_per_observation[step]):
                current_data.append(self.observations.get_band_data(step, 
                                                                    band))

            x_analysis, P_analysis, P_analysis_inverse, innovations = \
                self.do_all_bands(step, current_data, x_forecast, P_forecast,
                                  P_forecast_inverse, is_robust=is_robust)
            x_forecast = x_analysis*1.
            try:
                P_forecast = P_analysis*1.
            except:
                P_forecast = None
            try:
                P_forecast_inverse = P_analysis_inverse*1.
            except:
                P_forecast_inverse = None
                        
        return x_analysis, P_analysis, P_analysis_inverse            


    def do_all_bands(self, timestep, current_data, x_forecast, P_forecast,
                        P_forecast_inverse, convergence_tolerance=1e-3,
                        min_iterations=0, is_robust=False):
        not_converged = True
        # Linearisation point is set to x_forecast for first iteration
        x_prev = x_forecast*1.
        n_iter = 1
        n_bands = len(current_data)
        previous_convergence = 1e9
        while not_converged:
            Y = []
            MASK = []
            UNC = []
            META = []
            H_matrix = []
            for band, data in enumerate(current_data):
                # Create H0 and H_matrix around x_prev
                # Also extract single band information from nice package
                # this allows us to use the same interface as current
                # Deferring processing to a new solver method in solvers.py
                # Note that we could well do with passing `self.band_mapper`
                n_good_obs = np.sum(data.mask * self.state_mask)
                if n_good_obs > 0:
                    # Calculate H matrix & H0 vector around the linearisation
                    # point given by x_prev
                    H_matrix_= self._create_observation_operator(self.n_params,
                                                            data.emulator,
                                                            data.metadata,
                                                            data.mask,
                                                            self.state_mask,
                                                            x_prev,
                                                            band,
                                                            band_mapper = self.band_mapper)
                    if len(H_matrix_) == 2:
                        H_matrix.append(H_matrix_)
                    elif len(H_matrix_) == 3:
                        H_matrix.append([H_matrix_[0], H_matrix_[1]])
                    Y.append(data.observations)
                    MASK.append(data.mask)
                    UNC.append(data.uncertainty)
                    META.append(data.metadata)
                else:
                    LOG.info(f"Band {band+1:d} didn't have unmasked pixels")
            if len(Y) == 0:
                LOG.info("Couldn't find any usable (=unmasked) pixels."
                          "Not inverting anything. ")
                ### Must check what the innovations are used for further down
                return x_forecast, P_forecast, P_forecast_inverse, None
            
            # Now call the solver 
            x_analysis, P_analysis, P_analysis_inverse, \
                innovations, fwd_modelled = self.solver_multiband(
                    Y, MASK, H_matrix, x_prev, x_forecast,
                    P_forecast, P_forecast_inverse, UNC,
                    META)
            
            
            if n_iter > 1 and is_robust:
                
                # Mask out weird forecasts 
                MASK = robust_inflation(n_bands, innovations, Y, UNC,
                        self.state_mask, tolerance=6.)
            

            # Test convergence. We calculate the l2 norm of the difference
            # between the state at the previous iteration and the current one
            # There might be better tests, but this is quite straightforward
            #passer_mask = data.mask[self.state_mask]
            #maska = np.concatenate([passer_mask.ravel()
            #                        for i in range(self.n_params)])
            #convergence_norm = np.linalg.norm(x_analysis[maska] -
            #                                  x_prev[maska])/float(maska.sum())
            convergence_norm = np.linalg.norm(x_analysis - x_prev)/float(len(x_analysis))
            LOG.info(
                "Iteration # {:d}, convergence norm: {:g}".format(
                    n_iter, convergence_norm))
            if (convergence_norm < convergence_tolerance) and (
                    n_iter >= min_iterations):
                # Converged!
                LOG.info("Converged sensibly")
                not_converged = False
            elif previous_convergence < convergence_norm:
                # Too many iterations
                LOG.info("Diverging solution. Bailing out here")
                # Return previous solution
                LOG.info("--> Returning previous solution! :-0")
                x_analysis = x_prev*1.
                not_converged = False
                
            elif n_iter > 25:
                # Too many iterations
                LOG.warning("Bailing out after 25 iterations!!!!!!")
                not_converged = False

            # Update the linearisation point to the solution
            # after the current iteration...
            x_prev = x_analysis*1.
            previous_convergence = convergence_norm
            n_iter += 1
            
            INNOVATIONS = np.split(innovations, n_bands)
            # Need to plot the goodness of fit somewhere...
            import matplotlib.pyplot as plt
            M = []
            print(f"Iteration: {n_iter-1:d}")
            for band in range(n_bands):
                t = np.zeros_like(self.state_mask).astype(np.float)            
                t[self.state_mask*MASK[band]] = INNOVATIONS[band]
                M.append(t)
                print(f"\tBand: {band:d}, {np.mean(INNOVATIONS[band]):f}")
            print(f"Timestep {timestep.isoformat():s}")
            print(f"\tavg TLAI: {x_analysis[6::10].mean():f}")
            print(f"\tavg TCAB: {x_analysis[1::10].mean():f}")
            print(f"\tavg N: {x_analysis[0::10].mean():f}")
            plt.figure()
            t = np.zeros_like(self.state_mask).astype(np.float)
            t[self.state_mask] = -2*np.log(x_analysis[6::10])
            plt.imshow(t, interpolation="nearest", vmin=0,
                       vmax=5, cmap=plt.cm.inferno)
            plt.title(f"{timestep.isoformat():s} Iter: {n_iter-1:d}")
            plt.show()
            import ipdb;ipdb.set_trace()
            

        
        # Once we have converged...
        # Correct hessian for higher order terms
        #split_points = [m.sum( ) for m in MASK]
        
        ###try:
            #### SWitched off for time being
            ###INNOVATIONS = np.split(innovations, n_bands)
            #### Need to plot the goodness of fit somewhere...
            ###M = []
            ###for band in range(n_bands):
                ###t = np.zeros_like(self.state_mask).astype(np.float)
            
                ###t[self.state_mask*MASK[band]] = INNOVATIONS[band]
                ###M.append(t)

            ###P_correction = hessian_correction_multiband(data.emulator, x_analysis,
                                                    ###UNC, INNOVATIONS, MASK,
                                                    ###self.state_mask, n_bands,
                                                    ###self.n_params, self.band_mapper)
        ###except TypeError:
        P_correction = 0. # ;)
        P_analysis_inverse = P_analysis_inverse - P_correction
        
        # Done with this observation, move along...
        
        return x_analysis, P_analysis, P_analysis_inverse, innovations
                


    def solver_multiband(self, observations, mask, H_matrix, x0, x_forecast, P_forecast,
               P_forecast_inv, R_mat, the_metadata):
        
        x_analysis, P_analysis, P_analysis_inv, \
            innovations_prime, fwd_modelled = \
            variational_kalman_multiband(
                observations, mask, self.state_mask, R_mat, H_matrix,
                self.n_params, x0,
                x_forecast, P_forecast, P_forecast_inv, the_metadata,
                upper_bound=self.upper_bound, lower_bound=self.lower_bound)

        return x_analysis, P_analysis, P_analysis_inv, \
            innovations_prime, fwd_modelled
