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
from kafka.inference import create_nonlinear_observation_operator
from kafka.state_propagation import IdentityPropagator, NoPropagator


# Probably should be imported from somewhere else, but I can't see
# where from ATM... No biggy



        

###class DummyInferencePrior(_WrappingInferencePrior):
    ###"""
    ###This class is merely a dummy.
    ###"""

    ###def process_prior(self, parameters: List[str], time: Union[str, datetime], state_grid: np.array,



        
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
        print("Running ")
        run_kafka(roi, mask[this_Y:(this_Y+ny_valid), this_X:(this_X+nx_valid)],
                hex(chunk), time_grid, bhr_data, parameter_list)
            
def run_kafka(roi, mask, prefix, time_grid, bhr_data,
              parameter_list, the_prior=None):
    [ulx, uly, lrx, lry] = roi
    bhr_data.apply_roi(ulx, uly, lrx, lry)
    projection, geotransform = bhr_data.define_output()
    output = KafkaOutput(parameter_list, geotransform, projection,
                         "/data/selene/ucfajlg/tmp/",
                         prefix=prefix)

    if the_prior is None:
        the_prior = JRCPrior(parameter_list, mask)

    #Q = np.array([100., 100., 100., 100., 100., 100., 20.])
    Q = np.array([100., 1e5, 1e2, 100., 1e5, 1e2, 100.])
    
    band_mapper = [np.array([0, 1, 6, 2]),
                   np.array([3, 4, 6, 5])]
    
    state_propagator = IdentityPropagator(Q, 7, mask)
    #state_propagator = NoPropagator(Q, 7, mask)
    
    kf = LinearKalman(bhr_data, output, mask, 
                      create_nonlinear_observation_operator, parameter_list,
                      state_propagation=state_propagator.get_matrices,
                      prior=the_prior, band_mapper=band_mapper,
                      linear=False)

    # Get starting state... We can request the prior object for this
    x_forecast, P_forecast_inv = the_prior.process_prior(None)
        
    
    kf.run(time_grid, x_forecast, None, P_forecast_inv,
           iter_obs_op=True, is_robust=False)
    

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
    mask = np.zeros((2400,2400),dtype=np.bool8)
#    mask[900:940, 1300:1340] = True # Alcornocales
#    mask[640:700, 1400:1500] = True # Campinha
    mask[650:730, 1180:1280] = True # Arros
    #mask[ 2200:2395, 450:700 ] = True # Bondville, h11v04
    #mask = province_mask(provinces=["Córdoba"])

    bhr_data =  BHRObservations(emulator, tile, mcd43a1_dir, start_time,
                                end_time=None, mcd43a2_dir=None)
    base = datetime(2017,5,1)
    num_days = 180
    time_grid = []
    for x in range( 0, num_days, 16):
        time_grid.append( base + timedelta(days = x) )
        
    nx = ny = 2400
    them_chunks = [the_chunk for the_chunk in get_chunks(nx, ny, block_size= [256, 256])]
    
    try:
        from dask.distributed import Client                                                                                          
        #from distributed.deploy.ssh import SSHCluster
        #with open('./hosts.txt', 'rb') as f:
        #    hosts = f.read().split()

        #c = SSHCluster(scheduler_addr=hosts[0], scheduler_port = 8786, worker_addrs=hosts[1:], nthreads=0, nprocs=1,
        #               ssh_username=None, ssh_port=22, ssh_private_key=None, nohost=False, logdir='/tmp/')
        client = Client('tcp://tyche.geog.ucl.ac.uk:8786')
        A = client.map (wrapper, them_chunks)
        retval = client.gather(A)
    except OSError:
        retval = map(wrapper, them_chunks)
        list(retval)

    #from dask.distributed import Client
    #client=Client(scheduler_file="/home/ucfajlg/scheduler.json")
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


    
