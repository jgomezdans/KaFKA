#!/usr/bin/env python
import logging
logging.basicConfig( 
    level=logging.getLevelName(logging.DEBUG), 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename="the_log.log")
import os
from datetime import datetime, timedelta

import numpy as np

import kafka
from kafka.input_output import BHRObservations, KafkaOutput, get_chunks
from kafka import LinearKalman
from kafka.inference import block_diag
from kafka.inference import create_nonlinear_observation_operator
from kafka.state_propagation import IdentityPropagator, NoPropagator
from kafka.priors import JRCPrior

from InferenceInterface import kafka_inference

import gdal

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


##def launch_dask(hostfile):
    ##from dask.distributed import Client                                                                                          
    ##from distributed.deploy.ssh import SSHCluster
    ##with open(hostfile, 'rb') as f:
        ##hosts = f.read().split()
    ##c = SSHCluster(scheduler_addr=hosts[0], scheduler_port = 8786,
                   ##worker_addrs=hosts[1:], nthreads=0, nprocs=1,
                   ##ssh_username =None, ssh_port=22,
                   ##ssh_private_key=None, nohost=False, logdir='/tmp/')

    ##client = Client('tcp://tyche:8786')
    ##return client

if __name__ == "__main__":
    from dask.distributed import Client                                                                                          
    parameter_list = ["w_vis", "x_vis", "a_vis",
                     "w_nir", "x_nir", "a_nir", "TeLAI"]
    tile = "h17v05"
    start_time = "2017001"
    band_mapper = [np.array([0, 1, 6, 2]),
                   np.array([3, 4, 6, 5])]
    emulator = "./SAIL_emulator_both_500trainingsamples.pkl"
    
    if os.path.exists("/storage/ucfajlg/Ujia/MCD43/"):
        mcd43a1_dir = "/storage/ucfajlg/Ujia/MCD43/"
    else:
        mcd43a1_dir="/data/selene/ucfajlg/Ujia/MCD43"
    
    mask = province_mask(provinces=["Sevilla", "Córdoba"])
    #mask[900:940, 1300:1340] = True # Alcornocales
#    mask[640:700, 1400:1500] = True # Campinha
    #mask[650:730, 1180:1280] = True # Arros
    #mask[ 2200:2395, 450:700 ] = True # Bondville, h11v04
    #mask = province_mask(provinces=["Córdoba"])

    bhr_data =  BHRObservations(emulator, tile, mcd43a1_dir, start_time,
                                end_time=None, mcd43a2_dir=None)
    base = datetime(2017,5,1)
    num_days = 180
    time_grid = []
    for x in range( 0, num_days, 16):
        time_grid.append( base + timedelta(days = x) )
        
    
    Q = np.array([100., 1e5, 1e2, 100., 1e5, 1e2, 100.])
    state_propagator = IdentityPropagator(Q, 7, mask)
    prior = JRCPrior
    #dask_client = launch_dask("./hosts.txt")
    dask_client = Client('tcp://tyche:8786')
    output_files = kafka_inference(mask, time_grid, parameter_list,
                    bhr_data, prior, state_propagator,
                    "/data/selene/ucfajlg/tmp/", 
                    band_mapper, dask_client)





