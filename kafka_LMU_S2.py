#!/usr/bin/env python

import logging
logging.basicConfig( 
    level=logging.getLevelName(logging.DEBUG), 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename="LMU_S2.log")
import os
from datetime import datetime, timedelta
import numpy as np
import gdal

from kafka.inference import block_diag
from kafka import Sentinel2Observations
from kafka import NoPropagator, IdentityPropagator
from kafka import kafka_inference

class SAILPrior(object):
    def __init__ (self, parameter_list, state_mask):
        self.parameter_list = parameter_list
        if isinstance(state_mask, (np.ndarray, np.generic) ):
            self.state_mask = state_mask
        else:
            self.state_mask = self._read_mask(state_mask)
        # Cm  (g/cm2) mean=0.0140 std_dev=0.0030
        # Cw (g/cm2)  mean=0.017 std_dev=0.0032
        # Tcm = np.exp(-100*0.0140)
        # Tcw = np.exp(-50*0.017)
        self.mean = np.array([2.1, np.exp(-50./100.),
                                 np.exp(-7.0/100.), 0.1,
                                 np.exp(-50*0.017), np.exp(-100.*0.0140),
                                 np.exp(-4./2.), 70./90., 0.5, 0.9])
        sigma = np.array([0.001, 0.125,
                                 0.001, 0.2,
                                 0.05, 0.05,
                                 0.90, 0.001, 0.001, 0.001])
 
        
        self.covar = np.diag(sigma**2).astype(np.float32)
        self.covar[6, 1] = 0.4/np.sqrt(self.covar[1,1]*self.covar[6,6])
        self.covar[1, 6] = 0.4/np.sqrt(self.covar[1,1]*self.covar[6,6])
        self.covar[3, 1] = -0.98/np.sqrt(self.covar[1,1]*self.covar[3,3])
        self.covar[1, 3] = -0.98/np.sqrt(self.covar[1,1]*self.covar[3,3])

        self.inv_covar = np.diag(1./sigma**2).astype(np.float32)
        
    def _read_mask(self, fname):
        """Tries to read the mask as a GDAL dataset"""
        if not os.path.exists(fname):
            raise IOError("State mask is neither an array or a file that exists!")
        g = gdal.Open(fname)
        if g is None:
            raise IOError("{:s} can't be opened with GDAL!".format(fname))
        mask = g.ReadAsArray()
        return mask

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
            covar_list = [self.covar for m in range(n_pixels)]
            covar = block_diag(covar_list, dtype=np.float32)
            return x0, covar

#kafka_inference(mask, time_grid, parameter_list,
#                    observations, prior, propagator,
#                    output_folder, band_mapper, dask_client,
#                    chunk_size=[64, 64])


if __name__ == "__main__":
    # Read the state mask. Just a handful of fields near Munich
    g = gdal.Open("/home/ucfajlg/Data/python/LMU_testcase/" +
                  "data/ESU.tif")
    state_mask = g.ReadAsArray().astype(np.bool)
    
    # Parameter list 
    parameter_list = ['n', 'cab', 'car', 'cbrown', 'cw', 'cm',
                      'lai', 'ala', 'bsoil', 'psoil']
    
    s2_observations = Sentinel2Observations(
            "/home/ucfafyi/public_html/S2_data/32/U/PU/",
            "/home/ucfafyi/DATA/Multiply/emus/sail/",
            "/home/ucfajlg/Data/python/LMU_testcase/data/ESU.tif")

    base = datetime(2017,6,1)
    num_days = 120
    time_grid = []
    for x in range( 0, num_days, 5):
        time_grid.append( base + timedelta(days = x) )
        


    prior = SAILPrior
    
    
    Q = np.ones(len(parameter_list))*1e9
    Q[1] = 0.002
    Q[6] = 0.001
    Q[3] = 0.001
    Q[4] = 0.0005
    Q[5] = 0.0005
    #state_propagator = NoPropagator(Q, len(parameter_list), state_mask)
    state_propagator = IdentityPropagator(Q, len(parameter_list), state_mask)

    output_files = kafka_inference(state_mask, time_grid, parameter_list,
                    s2_observations, prior, state_propagator,
                    "/data/selene/ucfajlg/tmp_prop/", None, None,
                    chunk_size=[128, 128])
