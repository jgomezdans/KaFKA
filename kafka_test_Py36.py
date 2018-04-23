#!/usr/bin/env python

import logging
logging.basicConfig( 
    level=logging.getLevelName(logging.DEBUG), 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename="the_log.log")
import os
from datetime import datetime, timedelta

import numpy as np
import gdal
import osr
import scipy.sparse as sp

# from multiply.inference-engine blah blah blah
try:
    from multiply_prior_engine import PriorEngine
except ImportError:
    pass

import kafka
from kafka.input_output import BHRObservations, KafkaOutput
from kafka import LinearKalman
from kafka.inference import block_diag
from kafka.inference import propagate_information_filter_LAI
from kafka.inference import no_propagation
from kafka.inference import create_nonlinear_observation_operator


# Probably should be imported from somewhere else, but I can't see
# where from ATM... No biggy

def reproject_image(source_img, target_img, dstSRSs=None):
    """Reprojects/Warps an image to fit exactly another image.
    Additionally, you can set the destination SRS if you want
    to or if it isn't defined in the source image."""
    g = gdal.Open(target_img)
    geo_t = g.GetGeoTransform()
    x_size, y_size = g.RasterXSize, g.RasterYSize
    xmin = min(geo_t[0], geo_t[0] + x_size * geo_t[1])
    xmax = max(geo_t[0], geo_t[0] + x_size * geo_t[1])
    ymin = min(geo_t[3], geo_t[3] + y_size * geo_t[5])
    ymax = max(geo_t[3], geo_t[3] + y_size * geo_t[5])
    xRes, yRes = abs(geo_t[1]), abs(geo_t[5])
    if dstSRSs is None:
        dstSRS = osr.SpatialReference()
        raster_wkt = g.GetProjection()
        dstSRS.ImportFromWkt(raster_wkt)
    else:
        dstSRS = dstSRSs
    g = gdal.Warp('', source_img, format='MEM',
                  outputBounds=[xmin, ymin, xmax, ymax], xRes=xRes, yRes=yRes,
                  dstSRS=dstSRS)
    return g



###class DummyInferencePrior(_WrappingInferencePrior):
    ###"""
    ###This class is merely a dummy.
    ###"""

    ###def process_prior(self, parameters: List[str], time: Union[str, datetime], state_grid: np.array,


class JRCPrior(object):
    """Dummpy 2.7/3.6 prior class following the same interface as 3.6 only
    version."""

    def __init__ (self, parameter_list, state_mask):
        """It makes sense to have the list of parameters and state mask
        defined at this point, as they won't change during processing."""
        self.mean, self.covar, self.inv_covar = self._tip_prior() 
        self.parameter_list = parameter_list
        if isinstance(state_mask, (np.ndarray, np.generic) ):
            self.state_mask = state_mask
        else:
            self.state_mask = self._read_mask(state_mask)
            
    def _read_mask(self, fname):
        """Tries to read the mask as a GDAL dataset"""
        if not os.path.exists(fname):
            raise IOError("State mask is neither an array or a file that exists!")
        g = gdal.Open(fname)
        if g is None:
            raise IOError("{:s} can't be opened with GDAL!".format(fname))
        mask = g.ReadAsArray()
        return mask

    def _tip_prior(self):
        """The JRC-TIP prior in a convenient function which is fun for the whole
        family. Note that the effective LAI is here defined in transformed space
        where TLAI = exp(-0.5*LAIe).

        Returns
        -------
        The mean prior vector, covariance and inverse covariance matrices."""
        # broadly TLAI 0->7 for 1sigma
        sigma = np.array([0.12, 0.7, 0.0959, 0.15, 1.5, 0.2, 0.5])
        x0 = np.array([0.17, 1.0, 0.1, 0.7, 2.0, 0.18, np.exp(-0.5*2.)])
        # The individual covariance matrix
        little_p = np.diag(sigma**2).astype(np.float32)
        little_p[5, 2] = 0.8862*0.0959*0.2
        little_p[2, 5] = 0.8862*0.0959*0.2

        inv_p = np.linalg.inv(little_p)
        return x0, little_p, inv_p

    def process_prior ( self, time, inv_cov=True):
        # Presumably, self._inference_prior has some method to retrieve 
        # a bunch of files for a given date...
        n_pixels = self.state_mask.sum()
        x0 = np.array([self.mean for i in range(n_pixels)]).flatten()
        if inv_cov:
            inv_covar_list = [self.inv_covar for m in range(n_pixels)]
            inv_covar = block_diag(inv_covar_list, dtype=np.float32)
            return x0, inv_covar
        else:
            covar_list = [self.covar for m in xrange(n_pixels)]
            covar = block_diag(covar_list, dtype=np.float32)
            return x0, covar
        
class KafkaOutputMemory(object):
    """A very simple class to output the state."""
    def __init__(self, parameter_list):
        self.parameter_list = parameter_list
        self.output = {}
    def dump_data(self, timestep, x_analysis, P_analysis, P_analysis_inv,
                state_mask):
        solution = {}
        for ii, param in enumerate(self.parameter_list):
            solution[param] = x_analysis[ii::7]
        self.output[timestep] = solution


if __name__ == "__main__":
    
    parameter_list = ["w_vis", "x_vis", "a_vis",
                     "w_nir", "x_nir", "a_nir", "TeLAI"]
    
    tile = "h11v04"
    start_time = "2006185"
    
    emulator = "./SAIL_emulator_both_500trainingsamples.pkl"
    
    if os.path.exists("/storage/ucfajlg/Ujia/MCD43/"):
        mcd43a1_dir = "/storage/ucfajlg/Ujia/MCD43/"
    else:
        mcd43a1_dir="/data/MODIS/h11v04/MCD43"
    ####tilewidth = 75
    ###n_pixels = tilewidth*tilewidth
    mask = np.zeros((2400,2400),dtype=np.bool8)
    #mask[900:940, 1300:1340] = True # Alcornocales
    #mask[640:700, 1400:1500] = True # Campinha
    #mask[650:730, 1180:1280] = True # Arros
    mask[ 2200:2395, 450:700 ] = True # Bondville, h11v04

    bhr_data =  BHRObservations(emulator, tile, mcd43a1_dir, start_time,
                                end_time=None, mcd43a2_dir=None)

    projection, geotransform = bhr_data.define_output()

    output = KafkaOutput(parameter_list, geotransform, projection, "/tmp/")

    the_prior = JRCPrior(parameter_list, mask)
    
    kf = LinearKalman(bhr_data, output, mask, 
                      create_nonlinear_observation_operator,parameter_list,
                      state_propagation=None,
                      prior=the_prior,
                      linear=False)

    # Get starting state... We can request the prior object for this
    x_forecast, P_forecast_inv = the_prior.process_prior(None)
    
    Q = np.zeros_like(x_forecast)
    Q[6::7] = 0.025
    
    kf.set_trajectory_model()
    kf.set_trajectory_uncertainty(Q)
    
    base = datetime(2006,7,4)
    num_days = 180
    time_grid = []
    for x in range( 0, num_days, 16):
        time_grid.append( base + timedelta(days = x) )

    kf.run(time_grid, x_forecast, None, P_forecast_inv, iter_obs_op=True)
    
