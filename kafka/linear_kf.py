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
#from scipy.spatial.distance import squareform, pdist


from utils import  matrix_squeeze, spsolve2, reconstruct_array
from utils import locate_in_lut
from solvers import variational_kalman

# Set up logging
import logging
LOG = logging.getLogger(__name__+".linear_kf")


__author__ = "J Gomez-Dans"
__copyright__ = "Copyright 2017 J Gomez-Dans"
__version__ = "1.0 (09.03.2017)"
__license__ = "GPLv3"
__email__ = "j.gomez-dans@ucl.ac.uk"

Metadata = namedtuple('Metadata', 'mask uncertainty')
Previous_State = namedtuple("Previous_State", 
                            "timestamp x_vect cov_m icov_mv")

def run_emulator (gp, x, tol=None):
    # We select the unique values in vector x
    # Note that we could have done this using e.g. a histogram
    # or some other method to select solutions "close enough"
    unique_vectors = np.vstack({tuple(row) for row in x})
    if len(unique_vectors) == 1: # Prior!
        cluster_labels = np.zeros(x.shape[0], dtype=np.int16)
    elif len(unique_vectors) > 5e3:
        
        LOG.info("Clustering parameter space")
        mean = np.mean(x, axis=0) # 7 dimensions
        cov = np.cov(x, rowvar=0) # 7 x 7 dimensions
        # Draw a 300 element LUT
        unique_vectors = np.random.multivariate_normal( mean, cov, 
                                                       1500)
        # Assign each element of x to a LUT/cluster entry
        cluster_labels = locate_in_lut(unique_vectors, x)
    # Runs emulator for emulation subset
    try:
        H_, dH_ = gp.predict( unique_vectors, do_unc=False)
    except ValueError:
        # Needed for newer gp version
        H_, _, dH_ = gp.predict( unique_vectors, do_unc=False)
    
    H = np.zeros(x.shape[0])
    dH = np.zeros_like(x)
    try:
        nclust = cluster_labels.shape
    except NameError:
        for i, uniq in enumerate(unique_vectors):
            passer = np.all ( x == uniq, axis=1)
            H[passer] = H_[i]
            dH[passer, :] = dH_[i, :]
        return H, dH
    
    for label in np.unique(cluster_labels):
        H[cluster_labels==label] = H_[label]
        dH[cluster_labels==label, :] = dH_[label, :]
    return H, dH

def create_uncertainty(uncertainty, mask):
    """Creates the observational uncertainty matrix. We assume that
    uncertainty is a single value and we return a diagonal matrix back.
    We present this diagonal **ONLY** for pixels that have observations
    (i.e. not masked)."""
    good_obs = mask.sum()
    R_mat = np.ones (good_obs)*uncertainty*uncertainty
    return sp.dia_matrix((R_mat, 0), shape=(R_mat.shape[0], R_mat.shape[0]))

def create_linear_observation_operator (obs_op, n_params, metadata, 
                                        mask, x_forecast, band=None):
    """A simple **identity** observation opeartor. It is expected that you
    subclass and redefine things...."""
    good_obs = mask.sum() # size of H_matrix
    H_matrix = sp.dia_matrix(np.eye (good_obs))
    return H_matrix

def create_nonlinear_observation_operator(n_params, emulator, metadata,
                                          mask, x_forecast, band):
    """Using an emulator of the nonlinear model around `x_forecast`.
    This case is quite special, as I'm focusing on a BHR SAIL
    version (or the JRC TIP), which have spectral parameters
    (e.g. leaf single scattering albedo in two bands, etc.). This
    is achieved by using the `state_mapper` to select which bits
    of the state vector (and model Jacobian) are used."""
    LOG.info("Creating the ObsOp for band %d" % band)
    n_times = x_forecast.shape[0] / n_params
    good_obs = mask.sum()

    H_matrix = sp.lil_matrix((n_times, n_params * n_times),
                             dtype=np.float32)

    H0 = np.zeros(n_times, dtype=np.float32)

    # So the model has spectral components.
    if band == 0:
        # ssa, asym, TLAI, rsoil
        state_mapper = np.array([0, 1, 6, 2])
    elif band == 1:
        # ssa, asym, TLAI, rsoil
        state_mapper = np.array([3, 4, 6, 5])
        
    # This loop can be JIT'ed
    x0 = np.zeros((n_times, 4))
    for i in xrange(n_times):
        if mask.ravel()[i]:
            x0[i, :] = x_forecast[state_mapper + n_params * i]
    LOG.info("Running emulators")
    # Calls the run_emulator method that only does different vectors
    # It might be here that we do some sort of clustering
    H0_, dH = run_emulator(emulator, x0[mask.ravel()])
    n = 0
    LOG.info("Storing emulators in H matrix")
    # This loop can be JIT'ed too
    for i in xrange(n_times):
        if mask.ravel()[i]:
            H_matrix[i, state_mapper + n_params * i] = dH[n]
            H0[i] = H0_[n]
            n += 1
    LOG.info("\tDone!")

    return (H0, H_matrix.tocsr())

class LinearKalman (object):
    """The main Kalman filter class operating in raster data sets. Note that the
    goal of this class is not to consider complex, time evolving models, but
    rather grotty "0-th" order models!"""
    def __init__(self, observations, output,
                 linear=True, n_params=1, diagnostics=True, bands_per_observation=1):
        """The class creator takes a list of observations, some metadata and a
        pointer to an output array."""
        self.n_params = n_params
        self.observations = observations
        self.output = output
        self.diagnostics = diagnostics
        self.bands_per_observation = bands_per_observation
        if linear:
            self._create_observation_operator = create_linear_observation_operator
        else:
            self._create_observation_operator = create_nonlinear_observation_operator
        
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


    def advance(self, x_analysis, P_analysis, P_analysis_inverse):
        """Advance the state. If we have to update the state
        precision matrix, I use the information filter formalism.
        """
        # Needs to deal with the inverse analysis matrix ;-/
        x_forecast = self.trajectory_model.dot(x_analysis)
        if sp.issparse(self.trajectory_uncertainty) and P_analysis is not None:
            P_forecast = P_analysis + self.trajectory_uncertainty
            P_forecast_inverse = None
        elif sp.issparse(self.trajectory_uncertainty) and P_analysis is None:
            LOG.info("Updating prior *inverse covariance*")
            # These is an approximation to the information filter equations
            #(see e.g. Terejanu's notes)
            M = P_analysis_inverse # for convenience and to stay with 
                                   # Terejanu's notation
            # Main assumption here is that the "inflation" factor is
            # calculated using the main diagonal of M
            PQ_matrix = (np.ones(M.shape[0]) + (1./(M.diagonal())*
                                    self.trajectory_uncertainty.diagonal()))
            # Update P_f = P_a^{-1}/(I+P_a^{-1}.diag + Q)
            P_forecast_inverse = M*sp.dia_matrix((PQ_matrix,0), 
                                                 shape=M.shape)
            
            P_forecast = None
            
        else:
            trajectory_uncertainty = sp.dia_matrix((self.trajectory_uncertainty,
                                                    0), shape=P_analysis.shape)
            P_forecast = P_analysis + trajectory_uncertainty
            P_forecast_inverse = None

        return x_forecast, P_forecast, P_forecast_inverse
    
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
        is_first = True
        
        for ii,timestep in enumerate(time_grid):
            # First locate all available observations for time step of interest.
            # Note that there could be more than one...
            locate_times = [i for i, x in enumerate(self.observations.dates)
                        if x == timestep]
            self.current_timestep = timestep
            temp_times = [ self.observations.dates[k] for k in locate_times]
            locate_times = temp_times
            LOG.info("timestep %s" % timestep.strftime("%Y-%m-%d"))
            
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
                LOG.info("# of Observations: %d" % len(locate_times))

                x_analysis, P_analysis, P_analysis_inverse = self.assimilate (
                                     locate_times, x_forecast, P_forecast,
                                     P_forecast_inverse,
                                     approx_diagonal=approx_diagonal,
                                     refine_diag=refine_diag,
                                     iter_obs_op=iter_obs_op,
                                     is_robust=is_robust, diag_str=diag_str)

            self.output.dump_data(timestep, x_analysis, P_analysis, 
                              P_analysis_inverse)

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
                        if self.bands_per_observation == 1:
                            observations, R_mat, mask, the_metadata, emulator \
                                = self._get_observations_timestep(step, None)
                        else:
                            observations, R_mat, mask, the_metadata, emulator \
                                = self._get_observations_timestep(step, band)
                        cached_obs.append ( (observations, R_mat, mask, 
                                           the_metadata, emulator) )
                        print step, mask.sum()
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
                    observations, R_mat, mask, the_metadata, the_emulator  = \
                        cached_obs[band]
                    
                    if self.diagnostics:
                        LOG.info("Setting up diagnostics...")
                        plot_object = self._set_plot_view(diag_str, step, 
                                                          observations)
                        self._plotter_iteration_start(plot_object, x_forecast,
                                                      observations, mask )
                    if self.bands_per_observation == 1:
                        # Remember that x_prev is the value that the iteration
                        # is working on. Starts with x_forecast, but updated
                        H_matrix = self._create_observation_operator(
                            self.n_params, the_emulator, the_metadata, 
                            mask, x_prev, None)
                    else:
                        H_matrix = self._create_observation_operator(
                            self.n_params, the_emulator, the_metadata, 
                            mask, x_prev, band)
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
                else:
                    break

                if converged and n_iter > 1:
                    # Convergence, getting out of loop
                    # Store current state as x_forecast in case more obs today
                    # the analysis becomes the forecast for the next
                    # iteration
                    
                    break

                if n_iter >= 8:
                    # Break if we go over 10 iterations
                    LOG.info("Wow, too many iterations (%d)!"%n_iter)
                    LOG.info("Stopping iterations here")
                    converged = True
                    break
            if is_robust and converged:
                # TODO update mask using innovations
                pass
            
            if self.diagnostics and have_obs:
                LOG.info("Plotting")
                self._plotter_iteration_end(plot_object, x_analysis,
                                        P_analysis, innovations_prime, mask)

        # Store the current state as previous state
        # Rationale is that when called on demand, we need to be able to 
        # propagate state to next available observation
        self.previous_state = Previous_State(step, x_analysis,
                                            P_analysis, P_analysis_inverse)

        return x_analysis, P_analysis, P_analysis_inverse


    def solver(self, observations, mask, H_matrix, x_forecast, P_forecast,
                P_forecast_inv, R_mat, the_metadata):
        
        x_analysis, P_analysis, P_analysis_inv, innovations_prime = \
            variational_kalman (
            observations, mask, R_mat, H_matrix, self.n_params, x_forecast,
            P_forecast, P_forecast_inv, the_metadata)
        
        return x_analysis, P_analysis, P_analysis_inv, innovations_prime
