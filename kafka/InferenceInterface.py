#!/usr/bin/env python
from copy import deepcopy
from functools import partial
from pathlib import Path
from collections import namedtuple

from osgeo import gdal


from kafka.input_output import  get_chunks, KafkaOutput
from kafka import LinearKalman
from kafka.inference import create_nonlinear_observation_operator

def stitch_outputs(output_folder, parameter_list):
    # Get the output folder
    p = Path(output_folder)
    # Loop over parameters and find all the files for all the 
    # chunks and dates
    output_tiffs = {}
    for parameter in parameter_list:
        files = [fich for fich in p.glob(f"{parameter:s}*.tif")]
        dates = [fich.stem.split(parameter)[1].split("_")[1] 
                 for fich in files]
        fnames = []
        # Now for each data, stitch up all the chunks for that parameter
        for date in dates:
            sel_files = [fich.as_posix() 
                         for fich in files if fich.stem.find(date) >= 0 ]
            dst_ds = gdal.BuildVRT((p/f"{parameter:s}_{date:s}.vrt").as_posix(),
                                sel_files)
            fnames.append(dst_ds.GetDescription())
        # Potentially, create a multiband VRT/GTiff with all the dates?
        dst_ds = gdal.BuildVRT((p/f"{parameter:s}.vrt").as_posix(),
                               fnames,options=gdal.BuildVRTOptions(separate=True))
        dst_ds = gdal.Translate((p/f"{parameter:s}.tif").as_posix(),
                                (p/f"{parameter:s}.vrt").as_posix(),
                                options=gdal.TranslateOptions(format="GTiff",
                                                             creationOptions=["TILED=YES",
                                                                            "COMPRESS=DEFLATE"]))
        output_tiffs[parameter] = dst_ds.GetDescription()
    return output_tiffs
        

def chunk_inference(roi, prefix, current_mask, configuration):

    [ulx, uly, lrx, lry] = roi
    
    configuration.observations.apply_roi(ulx, uly, lrx, lry)
    projection, geotransform = configuration.observations.define_output()
    output = KafkaOutput(configuration.parameter_list, 
                         geotransform, projection,
                         configuration.output_folder,
                         prefix=prefix)

    #Q = np.array([100., 1e5, 1e2, 100., 1e5, 1e2, 100.])
    ## state_propagator = IdentityPropagator(Q, 7, mask)
    the_prior = configuration.prior(configuration.parameter_list,
                                    current_mask)
    state_propagator = deepcopy(configuration.propagator)
    state_propagator.mask = current_mask
    state_propagator.n_elements = current_mask.sum()
    kf = LinearKalman(configuration.observations, 
                      output, current_mask, 
                      create_nonlinear_observation_operator, 
                      configuration.parameter_list,
                      state_propagation=state_propagator.get_matrices,
                      prior=the_prior, 
                      band_mapper=configuration.band_mapper,
                      linear=False)

    # Get starting state... We can request the prior object for this
    x_forecast, P_forecast_inv = the_prior.process_prior(None)
        
    
    kf.run(configuration.time_grid, x_forecast, None, P_forecast_inv,
           iter_obs_op=True, is_robust=False)


def chunk_wrapper(the_chunk, config):
    this_X, this_Y, nx_valid, ny_valid, chunk = the_chunk
    ulx = this_X
    uly = this_Y
    lrx = this_X + nx_valid
    lry = this_Y + ny_valid
    
    roi = [ulx, uly, lrx, lry]
    
    if config.mask[this_Y:(this_Y+ny_valid), this_X:(this_X+nx_valid)].sum() > 0:   
        print("Running chunk %s" % ( hex(chunk)))
        chunk_inference(roi, hex(chunk), 
                        config.mask[this_Y:(this_Y+ny_valid),
                                    this_X:(this_X+nx_valid)],
                        config)
        return hex(chunk)
            


def kafka_inference(mask, time_grid, parameter_list,
                    observations, prior, propagator,
                    output_folder, band_mapper, dask_client=None,
                    chunk_no=None, chunk_size=[128, 128]):
    
    # First, put the configuration in its own object to minimise
    # variable transport
    
    Config = namedtuple("Config", ["mask", "time_grid", "parameter_list",
                                   "observations", "prior", "propagator",
                                   "output_folder", "band_mapper"])    
    config = Config(mask, time_grid, parameter_list, observations,
                    prior, propagator, output_folder, band_mapper)
    nx, ny = mask.shape
    them_chunks = [the_chunk for the_chunk in get_chunks(nx, ny,
                    block_size=chunk_size)]
    
    wrapper = partial(chunk_wrapper, config=config)

    if dask_client is None:
        if chunk_no is None:
            chunk_names = list(map(wrapper, them_chunks))
        else:
            #chunk_name = wrapper(chunk_no)
            print(hex(chunk_no))
            return hex(chunk_no)#chunk_name
    else:
        A = dask_client.map (wrapper, them_chunks)
        retval = dask_client.gather(A)
    
    return stitch_outputs(output_folder, parameter_list)

    

    
    
