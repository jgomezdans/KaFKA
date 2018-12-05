#!/usr/bin/env python
"""

"""
import logging

import numpy as np

import scipy.sparse as sp

LOG = logging.getLogger(__name__)

def sar_observation_operator(x, theta, polarisation):

    """
    For the sar_observation_operator a simple Water Cloud Model (WCM) is used
    We assume that the WCM is given by

    -----------------------------------------------------
    tau = exp(-2*B*V/cos(theta))
    sigma_veg = A*V**E*cos(theta)*(1-tau)
    sigma_soil = 10**((C+D*SM)/10)
    sigma_0 = sigma_veg + tau*sigma_soil

    A, B, C, D, E are parameters determined by fitting of the model
    V is a vegetation descriptor e.g. LAI, LWAI, VWM
    SM is the soil moisture [m^3/m^3]
    sigma_veg is the volume component of the backscatter [m^3/m^3]
    sigma_soil is the surface component of the backscatter [m^3/m^3]
    tau is the two way attenuation through the canopy (unitless)
    sigma_0 is the total backscatter in [m^3/m^3]
    ----------------------------------------------------

    Input
    -----
    polarisation: considered polarisation as string
    x: 2D array where every row is the set of parameters for one pixel

    Output
    ------
    sigma_0: predicted backscatter for each individual parameter set
    grad: gradient for each individual parameter set and each parameter determined by 2D input array x. The gradient should thus have the same size and shape as x
    sigma_veg: predicted volume component of sigma_0
    sigma_surf: predicted surface component of sigma_0
    tau: predicted two-way attenuation through the canopy
    """

    # x 2D array where every row is the set of parameters for one pixel
    x = np.atleast_2d(x)

    # conversion of incidence angle to radiant
    # the incidence angle itself should probably implemented in x)
    # TODO needs to come from the data
    theta = np.deg2rad(theta.mean())

    # Simpler definition of cosine of theta
    # Averaged over scene.
    mu = np.cos(theta)
    
    # the model parameters (A, B, C, D, E) for different polarisations
    parameters = {'VV': [0.0846, 0.0615, -14.8465, 15.907, 1.],
                  'VH': [0.0795, 0.1464, -14.8332, 15.907, 0.]}
    # Select model parameters
    try:
        A, B, C, D, E = parameters[polarisation]
    except KeyError:
        raise ValueError('Only VV and VH polarisations available!')

    if np.any(x[:, 0] <= 0.):
        raise ValueError("Negative LAI!")
    if np.any(x[:, 1] <= 0.):
        raise ValueError("Negative SM!")

    # Calculate Model
    tau = np.exp(-2. * B / mu * x[:, 0])
    sigma_veg = A * np.power(x[:, 0], E) * mu * (1. - tau)
    sigma_surf = 10. ** ((C + D * x[:, 1]) / 10.)

    sigma_0 = sigma_veg + tau * sigma_surf


    # Calculate Gradient (grad has same dimension as x)
    grad = x*0
    n_elems = x.shape[0]
    for i in range(n_elems):
        z = np.power(x[i, 0], E)
        if np.isnan(z):
            z = 1.
        z1 = np.power(x[i, 0], E-1.)
        if np.isnan(z1):
            z1 = 1.
        tau_value = np.exp(-2. * B / mu * x[i, 0])
        grad[i, 0] = A * E * mu * z1 * (1. - tau_value) + \
            2. * A * B * z * tau_value - (
            (2. ** (1/10. * (C + D * x[i, 1]) + 1.)) *
            (5. ** (1/10. * (C + D * x[i, 1])) * B * tau_value)
            ) / mu
        grad[i, 1] = D * np.log(10.) * tau_value * 10. ** (
            1./10. * (C + D * x[i, 1]) - 1.)

    # returned values are linear scaled not dB!!!
    # return sigma_0, grad, sigma_veg, sigma_surf, tau
    if np.any(np.isnan(sigma_0)):
        raise ValueError("Groan!")
    if np.any(np.isnan(grad)):
        raise ValueError("More Groan!")
    return sigma_0, grad


def create_sar_observation_operator(n_params, forward_model, metadata,
                                    mask, state_mask,  x_forecast, band,
                                    band_mapper=None):
    """Creates the SAR observation operator using the Water Cloud SAR forward
    model (defined above).

    Parameters
    -----------
    n_params: int
        Number of parameters in the state vector per pixel
    forward_model: function
        The function to call the forward model. Defined above
    metadata: list
        Not used
    mask: array
        A 2D mask with the observational valid pixels
    state_mask: array
        A 2D mask with the pixels that will be used for state inference
    x_forecast: array
        The state vector around which the linearisation will be done. Order is
        parameters per pixel (e.g. LAI1, SM1, LAI2, SM2, ..., LAIN, SMN)
    band: array
        The band number. Assumed 0 is VV and 1 is VH.

    Returns
    --------
    H0, dH
    """
    LOG.info("Creating the ObsOp for band %d" % band)
    n_times = x_forecast.shape[0] // n_params
    # good_obs = mask.sum()
    H_matrix = sp.lil_matrix((n_times, n_params * n_times),
                             dtype=np.float32)
    H0 = np.zeros(n_times, dtype=np.float32)

    # So the model has spectral components.
    if band == 0:
        # VV
        polarisation = "VV"
    elif band == 1:
        # VH
        polarisation = "VH"
    # This loop can be JIT'ed
    x0 = np.zeros((n_times, n_params))
    theta = np.zeros((n_times))
    for i, m in enumerate(mask[state_mask].flatten()):
        if m:
            x0[i, :] = x_forecast[(n_params * i): (n_params*(i+1))]
            theta[i] = 23.#metadata['incidence_angle']
    LOG.info("Running SAR forward model")
    # Calls the run_emulator method that only does different vectors
    # It might be here that we do some sort of clustering

    H0_, dH = forward_model(x0[mask[state_mask]], theta, polarisation)

    LOG.info("Storing emulators in H matrix")
    # This loop can be JIT'ed too
    n = 0
    
    for i, m in enumerate(mask[state_mask].flatten()):
        if m:
            H_matrix[i, (n_params * i): (n_params*(i+1))] = dH[n]
            H0[i] = H0_[n]
            n += 1
    LOG.info("\tDone!")
    return (H0, H_matrix.tocsr())

    # # Calculate Gradient without conversion of sigma_soil from dB to linear
    # grad = x*0
    # n_elems = x.shape[0]
    # for i in range(n_elems):
    #     tau = np.exp(-2 * B / mu * x[i, 0])
    #     grad[i, 0] = 2 * A * B * x[i, 0] * tau - \
    #         A * mu * tau + \
    #         A * mu - \
    #         2 * B * C * tau / mu - \
    #         2 * B * D * x[i, 1] * tau / mu
    #     grad[i, 1] = D * tau
