#!/usr/bin/env python
"""Demo-es the linear KF case, and demonstrates how to add diagnostics."""

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

from linear_kf import LinearKalman, Metadata

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


class TestKalman(LinearKalman):
    """We used a derived class to add user-selected diagnostics"""
    def __init__(self, observations, observation_times,
                 observation_metadata, output_array, output_unc,
                 n_params=1, diagnostics=True, bands_per_observation=1):
        LinearKalman.__init__(self, observations, observation_times,
                              observation_metadata, output_array,
                              output_unc, n_params=n_params, diagnostics=True,
                              bands_per_observation=bands_per_observation)

    def _set_plot_view(self, diag_str, timestep, observations):
        obj = namedtuple("PlotObject", "fig axs fname nx ny")
        title = "%s %d" % (diag_str, timestep)
        fname = "diagnostic_%s_%04d" % ( diag_str, timestep)
        fig, axs = plt.subplots(nrows=self.n_params, ncols=5, sharex=True,
                               sharey=True, figsize=(15,5),
                                subplot_kw=dict(
                                    adjustable='box-forced', aspect='equal') )
        axs = axs.flatten()
        fig.suptitle(title)
        ny, nx = observations.shape
        plot_obj = obj(fig, axs, fname, nx, ny )
        return plot_obj

    def _plotter_iteration_start(self, plot_obj, x, obs, mask):
        cmap = plt.cm.viridis
        cmap.set_bad = "0.8"
        plot_obj.axs[0].imshow (x.reshape(obs.shape), interpolation='nearest',
                       cmap=cmap)
        plot_obj.axs[0].set_title("Prior state")
        plot_obj.axs[1].imshow(obs, interpolation='nearest',
                      cmap=cmap)
        plot_obj.axs[1].set_title("Observations")

    def _plotter_iteration_end(self, plot_obj, x, P, innovation, mask):
        cmap = plt.cm.viridis
        cmap.set_bad = "0.8"

        M = np.ones((plot_obj.ny, plot_obj.nx))*np.nan
        not_masked = mask.reshape((plot_obj.ny, plot_obj.nx))
        M[not_masked] = innovation
        plot_obj.axs[2].imshow (M, interpolation='nearest',
                       cmap=cmap)
        plot_obj.axs[2].set_title("Innovation")
        plot_obj.axs[3].imshow(x.reshape((plot_obj.ny, plot_obj.nx)),
                               interpolation='nearest', cmap=cmap)
        plot_obj.axs[3].set_title("Posterior mean")
        unc = P.diagonal().reshape((plot_obj.ny, plot_obj.nx))
        plot_obj.axs[4].imshow(np.sqrt(unc),
                               interpolation='nearest', cmap=cmap)
        plot_obj.axs[4].set_title("Posterior StdDev")

        plot_obj.fig.savefig(plot_obj.fname + ".png", dpi=72,
                         bbox_inches="tight")
        plt.close(plot_obj.fig)





if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    f = np.load("../test_identity_data.npz")
    data = f['data']
    qa = f['qa']
    time_steps = np.arange(0,100,2)
    QA = []
    for i,t in enumerate(time_steps):
        meta = Metadata(np.where(qa[i,:,:]==0, False, True), [0.1])
        QA.append (meta)

    output_array = np.zeros((100, 50, 50))
    output_unc = np.zeros((100, 50, 50))
    kalman_filter = TestKalman(data, time_steps, QA, output_array, output_unc)
    kalman_filter.set_trajectory_model(50, 50)
    kalman_filter.set_trajectory_uncertainty(0.005, 50,50)
    x_f = 0.5*np.ones(50*50).ravel()

    P_f = sp.eye(50*50, 50*50, format="csr", dtype=np.float32)


    kalman_filter.run(x_f, P_f, refine_diag=True)

