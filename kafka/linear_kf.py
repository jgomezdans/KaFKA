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
                 n_params=1):
        """The class creator takes a list of observations, some metadata and a
        pointer to an output array."""
        self.n_params = n_params
        self.observations = observations
        self.observation_times = observation_times
        self.metadata = observation_metadata
        self.output = output_array
        self.output_unc = output_unc

    def _dump_output(self, step, timestep, x_analysis, P_analysis):
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

    def set_trajectory_model(self):
        """In a Kalman filter, the state is progated from time `t` to `t+1`
        using a model. We assume that this model is a matrix, and for the time
        being, the matrix is the identity matrix. That's how we roll!"""
        n = self.observations.shape[1]*self.observations.shape[2]
        self.trajectory_model = sp.eye(self.n_params*n, self.n_params*n,
                                       format="csr")

    def set_trajectory_uncertainty(self, Q):
        """In a Kalman filter, the model that propagates the state from time
        `t` to `t+1` is assumed to be *wrong*, and this is indicated by having
        additive Gaussian noise, which we assume is zero-mean, and controlled by
        a covariance matrix `Q`. Here, you can provide the main diagonal of `Q`.

        Parameters
        -----------
        Q: array
            The main diagonal of the model uncertainty covariance matrix.
        """
        n = self.observations.shape[1]*self.observations.shape[2]
        self.trajectory_uncertainty = sp.eye(self.n_params*n, self.n_params*n,
                                       format="csr").dot(Q)

    def create_uncertainty(self, uncertainty, mask):
        """Creates the observational uncertainty matrix. We assume that
        uncertainty is a single value and we return a diagonal matrix back.
        We present this diagonal **ONLY** for pixels that have observations
        (i.e. not masked)."""
        good_obs = mask.sum()
        R_mat = np.ones (good_obs)*uncertainty*uncertainty
        return sp.dia_matrix((R_mat, 0), shape=(R_mat.shape[0], R_mat.shape[0]))

    def create_observation_operator (self, metadata, x_forecast):
        """A simple **identity** observation opeartor. It is expected that you
        subclass and redefine things...."""
        good_obs = metadata.mask.sum() # size of H_matrix
        H_matrix = sp.dia_matrix(np.eye (good_obs))
        return H_matrix

    def advance(self, x_analysis, P_analysis):
        """Advance the state"""
        x_forecast = self.trajectory_model.dot(x_analysis)
        if sp.issparse(self.trajectory_uncertainty):
            P_forecast = P_analysis + self.trajectory_uncertainty
        else:
            trajectory_uncertainty = sp.dia_matrix((self.trajectory_uncertainty,
                                                    0), shape=P_analysis.shape)
            P_forecast = P_analysis + trajectory_uncertainty

        return x_forecast, P_forecast

    def run(self, x_forecast, P_forecast,
                   band=None, approx_diagonal=True, refine_diag=True,
                   iter_obs_op=False, is_robust=False):
        is_first = True
        for ii,timestep in enumerate(np.arange(self.observation_times.min(),
                                  self.observation_times.max() + 1)):
            # First locate all available observations for time step of interest.
            # Note that there could be more than one...
            locate_times = [i for i, x in enumerate(self.observation_times)
                        if x == timestep]

            if not is_first:
                x_forecast, P_forecast = self.advance(x_analysis, P_analysis)
            is_first = False
            if len(locate_times) == 0:
                # Just advance the time
                continue
            else:
                # We do have data, so we assimilate
                x_analysis, P_analysis = self.assimilate (locate_times,
                                 x_forecast, P_forecast,
                                 band=band, approx_diagonal=approx_diagonal,
                                 refine_diag=refine_diag,
                                 iter_obs_op=iter_obs_op, is_robust=is_robust)

            self._dump_output(ii, timestep, x_analysis, P_analysis)

    def assimilate(self, locate_times, x_forecast, P_forecast,
                   band=None, approx_diagonal=True, refine_diag=False,
                   iter_obs_op=False, is_robust=False):
        """The method assimilates the observatins at timestep `timestep`, using
        a prior a multivariate Gaussian distribution with mean `x_forecast` and
        variance `P_forecast`."""
        #import pdb;pdb.set_trace()
        for step in locate_times:
            print step
            # Extract observations, mask and uncertainty for the current time
            observations, R_mat, mask, the_metadata = \
                self._get_observations_timestep(step, band)
            # The assimilation works if data is there, so we need to reduce the
            # rank of the matrices by ignoring the masked data. `matrix_squeeze`
            # helps with this...

            P_forecast_prime = matrix_squeeze(P_forecast, mask=mask,
                                              n_params=self.n_params)

            # MAIN ITERATION loop
            # In an EKF, we would iterate and update the observation operator
            # until convergence is reached. If the observation operator is
            # linear, then no iterations are needed.

            while True:
                H_matrix = self.create_observation_operator(the_metadata,
                                                              x_forecast )
                # At this stage, we have a forecast (prior), the observations
                # and the observation operator, so we proceed with the
                # assimilation
                if approx_diagonal:
                    # We approximate the inverse matrix by a division assuming
                    # P_forecast is diagonal

                    R_mat_prime = np.array(R_mat.diagonal()).squeeze()

                    S = (H_matrix.dot(P_forecast_prime)).dot(H_matrix.T) + R_mat
                    nn1, nn2 = S.shape
                    S_inv = sp.dia_matrix((
                        [1./np.array(S.diagonal()).squeeze()],[0]),
                        shape=(nn1, nn2))

                    kalman_gain = P_forecast_prime.dot(H_matrix.T).dot(S_inv)

                if refine_diag:
                    #P_forecast_prime = P_forecast_prime.todia()
                    ####S = H_matrix.dot(P_forecast_prime).dot(H_matrix.T) + R_mat
                    S = (H_matrix.T.dot(P_forecast_prime)).dot(H_matrix) + R_mat
                    S = S.tocsc()
                    XX = spsolve2(S.T, H_matrix).T # This might require some
                    # speedups...
                    kalman_gain1 = P_forecast_prime.dot (XX)


                x_forecast_prime = matrix_squeeze(x_forecast, mask=mask.ravel(),
                                                  n_params=self.n_params)
                x_analysis_prime = x_forecast_prime + \
                                   kalman_gain*(observations.ravel()[mask.ravel()] - \
                                       H_matrix.dot(x_forecast_prime))
                P_analysis_prime = ((sp.eye(kalman_gain.shape[0], kalman_gain.shape[0])
                               - kalman_gain*H_matrix)*P_forecast_prime)
                # Now move
                x_analysis = reconstruct_array ( x_analysis_prime, x_forecast,
                                                 mask.ravel(), n_params=self.n_params)
                small_diagonal = np.array(P_analysis_prime.diagonal()).squeeze()
                big_diagonal = np.array(P_forecast.diagonal()).squeeze()
                P_analysis_diag = reconstruct_array(small_diagonal, big_diagonal,
                                               mask, n_params=self.n_params)
                P_analysis = sp.dia_matrix ( (P_analysis_diag, 0),
                                             shape=P_forecast.shape)

                if iter_obs_op:
                    # TODO test for convergence
                    converged = True
                else:
                    converged = True
#                if is_robust and converged:
#                    break
#                    # TODO robust re-masking
#                    # We should have a robust mechanism that checks whether the state
#                    # is too far from the observations, and if so, flag them as
#                    # outliers
                if converged:
                    break
            return x_analysis, P_analysis


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    f = np.load("../test_identity_data.npz")
    data = f['data']
    qa = f['qa']
    time_steps = np.arange(0,100,2)
    QA = []
    for i,t in enumerate(time_steps):
        meta = Metadata(np.where(qa[i,:,:]==0, False, True), 0.1)
        QA.append (meta)

    output_array = np.zeros((100, 50, 50))
    output_unc = np.zeros((100, 50, 50))
    kalman_filter = LinearKalman(data, time_steps, QA, output_array, output_unc)
    kalman_filter.set_trajectory_model()
    kalman_filter.set_trajectory_uncertainty(0.005)
    x_f = 0.5*np.ones(50*50).ravel()

    P_f = sp.eye(50*50, 50*50, format="csr", dtype=np.float32)


    kalman_filter.run(x_f, P_f, refine_diag=True)

    fig, axs = plt.subplots ( 1, 3, sharex=True, sharey=True, figsize=(15,5))
    axs = axs.flatten()
    get_state = lambda i: kalman_filter.output[i]
    get_state_unc= lambda i: kalman_filter.output_unc[i]
    get_obs = lambda i: data[i/2] if i%2 == 0 else np.nan*np.ones((50,50))
    im1 = axs[1].imshow (get_state(0), animated=True, interpolation='nearest',
                     vmin=0, vmax=1)
    im2 = axs[2].imshow (get_state_unc(0), animated=True, interpolation='nearest',
                     vmin=0, vmax=0.1)

    im3 = axs[0].imshow (get_obs(0), animated=True, interpolation='nearest',
                     vmin=0, vmax=1)
    axs[1].set_title("State")
    axs[2].set_title("State uncertainty")
    axs[0].set_title("Observations")
    [axs[i].set_ylim(0, 50) for i in xrange(3)]
    [axs[i].set_xlim(0, 50) for i in xrange(3)]
    [axs[i].axis('off') for i in xrange(3)]
    plt.tight_layout()
    def updatefig (i):
        im1.set_array(get_state(i))
        im2.set_array(get_state_unc(i))
        im3.set_array(get_obs(i))

        return im1,im2,im3

    ani = animation.FuncAnimation(fig, updatefig, frames=np.arange(100),
                                  interval=100, blit=False)

    ani.save('state.gif', writer='imagemagick')
