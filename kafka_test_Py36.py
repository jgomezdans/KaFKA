#!/usr/bin/env python
import copy
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
from kafka.input_output import BHRObservations, KafkaOutput, get_chunks
from kafka import LinearKalman
from kafka.inference import block_diag
from kafka.inference import propagate_information_filter_LAI
from kafka.inference import no_propagation
from kafka.inference import create_nonlinear_observation_operator


# Probably should be imported from somewhere else, but I can't see
# where from ATM... No biggy




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
            covar_list = [self.covar for m in range(n_pixels)]
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

def get_chunks(nx, ny, block_size= [256, 256]):
    blocks = []
    nx_blocks = (int)((nx + block_size[0] - 1) / block_size[0])
    ny_blocks = (int)((ny + block_size[1] - 1) / block_size[1])
    nx_valid, ny_valid = block_size
    chunk_no = 0
    for X in range(nx_blocks):
        # change the block size of the final piece
        if X == nx_blocks - 1:
            nx_valid = nx - X * block_size[0]
            buf_size = nx_valid * ny_valid

        # find X offset
        this_X = X * block_size[0]

        # reset buffer size for start of Y loop
        ny_valid = block_size[1]
        buf_size = nx_valid * ny_valid

        # loop through Y lines
        for Y in range(ny_blocks):
            # change the block size of the final piece
            if Y == ny_blocks - 1:
                ny_valid = ny - Y * block_size[1]
                buf_size = nx_valid * ny_valid
            chunk_no += 1
            #if skip_chunks is not None and chunk_no < skip_chunks:
            #    continue
            # find Y offset
            this_Y = Y * block_size[1]
            yield this_X, this_Y, nx_valid, ny_valid, chunk_no


def wrapper(the_chunk):
    this_X, this_Y, nx_valid, ny_valid, chunk = the_chunk
    ulx = this_X
    uly = this_Y
    lrx = this_X + nx_valid
    lry = this_Y + ny_valid
    
    roi = [ulx, uly, lrx, lry]
    if mask[this_Y:(this_Y+ny_valid), this_X:(this_X+nx_valid)].sum() > 0:   
        run_kafka(roi, mask[this_Y:(this_Y+ny_valid), this_X:(this_X+nx_valid)],
                hex(chunk), time_grid, bhr_data, parameter_list)
            
def run_kafka(roi, mask, prefix, time_grid, bhr_data,
              parameter_list, the_prior=None):
    [ulx, uly, lrx, lry] = roi
    bhr_data.apply_roi(ulx, uly, lrx, lry)
    projection, geotransform = bhr_data.define_output()
    output = KafkaOutput(parameter_list, geotransform, projection, "/tmp/",
                         prefix=prefix)

    if the_prior is None:
        the_prior = JRCPrior(parameter_list, mask)
    
    kf = LinearKalman(bhr_data, output, mask, 
                      create_nonlinear_observation_operator, parameter_list,
                      state_propagation=None,
                      prior=the_prior,
                      linear=False)

    # Get starting state... We can request the prior object for this
    x_forecast, P_forecast_inv = the_prior.process_prior(None)
    
    Q = np.zeros_like(x_forecast)
    Q[6::7] = 0.025
    
    kf.set_trajectory_model()
    kf.set_trajectory_uncertainty(Q)
    
    
    kf.run(time_grid, x_forecast, None, P_forecast_inv, iter_obs_op=True)
    

def province_mask(raster='HDF4_EOS:EOS_GRID:"/data/selene//ucfajlg/Ujia/MCD43/' + 
                  'MCD43A1.A2017214.h17v05.006.2017223172048.hdf":MOD_Grid_BRDF:' +
                  'BRDF_Albedo_Parameters_Band2',
                  shp = "/vsizip/vsicurl/http://www2.geog.ucl.ac.uk/~ucfajlg" +
                  "/Provincias_ETRS89_30N.zip/Provincias_ETRS89_30N.shp",
                  provinces=["Sevilla", "Granada", "Córdoba","Cádiz", "Huelva"]):
    mask = np.zeros((2400, 2400), dtype=np.bool)
    field="Texto"
    for prov in provinces:
        g = gdal.Warp("", raster, format="MEM",
                cutlineDSName = shp,
                cutlineWhere=f"{field:s} ='{prov:s}'",
                dstNodata = 0)
        data = g.GetRasterBand(1).ReadAsArray()
        ok_data = data != 0
        mask[ok_data] = True
    return mask


if __name__ == "__main__":
    
    parameter_list = ["w_vis", "x_vis", "a_vis",
                     "w_nir", "x_nir", "a_nir", "TeLAI"]
    
    tile = "h17v05"
    start_time = "2017001"
    
    emulator = "./SAIL_emulator_both_500trainingsamples.pkl"
    
    if os.path.exists("/storage/ucfajlg/Ujia/MCD43/"):
        mcd43a1_dir = "/storage/ucfajlg/Ujia/MCD43/"
    else:
        mcd43a1_dir="/data/selene/ucfajlg/Ujia/MCD43"
    ####tilewidth = 75
    ###n_pixels = tilewidth*tilewidth
#    mask = np.zeros((2400,2400),dtype=np.bool8)
#    mask[900:940, 1300:1340] = True # Alcornocales
#    mask[640:700, 1400:1500] = True # Campinha
#    mask[650:730, 1180:1280] = True # Arros
    #mask[ 2200:2395, 450:700 ] = True # Bondville, h11v04
    mask = province_mask()

    bhr_data =  BHRObservations(emulator, tile, mcd43a1_dir, start_time,
                                end_time=None, mcd43a2_dir=None)
    base = datetime(2017,1,1)
    num_days = 360
    time_grid = []
    for x in range( 0, num_days, 16):
        time_grid.append( base + timedelta(days = x) )
        
    nx = ny = 2400
    them_chunks = [the_chunk for the_chunk in get_chunks(nx, ny, block_size= [256, 256])]
    from dask.distributed import Client
    client=Client(scheduler_file="/home/ucfajlg/scheduler.json")
    A = client.map (wrapper, them_chunks)
    retval = client.gather(A)
    #B = client.submit(A)
    #total = B.result()
    
    #for the_chunk in get_chunks(nx, ny, block_size= [256, 256]):
        #this_X, this_Y, nx_valid, ny_valid, chunk = the_chunk
        #ulx = this_X
        #uly = this_Y
        #lrx = this_X + nx_valid
        #lry = this_Y + ny_valid
        
        #roi = [ulx, uly, lrx, lry]
        #if mask[this_Y:(this_Y+ny_valid), this_X:(this_X+nx_valid)].sum() > 0:   
            #run_kafka(roi, mask[this_Y:(this_Y+ny_valid), this_X:(this_X+nx_valid)],
                    #hex(chunk), time_grid, bhr_data, parameter_list)
            #print(ulx, uly, lrx, lry, hex(chunk))   


    
