#!/usr/bin/env python
"""
A script that averages the MODIS MCD43 (C5) kernels over a
period of time, saving to a GeoTIFF the mean kernel weight
value over the period, as well as the standard deviation, and 
the number of samples that went into the calculation.
"""
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

__author__ = "J Gomez-Dans"
__copyright__ = "Copyright 2017 J Gomez-Dans"
__version__ = "1.0 (09.03.2017)"
__license__ = "GPLv3"
__email__ = "j.gomez-dans@ucl.ac.uk"


import sys
import argparse
import os
import glob

import numpy as np
import gdal

def read_mcd43(band, fname_a1, fname_a2):
    """
    Reads the MCD43A1 and A2 files, mask the elements and return
    both the masked kernel parameters, as the information to create
    a similar geo dataset.
    """
    
    fname = 'HDF4_EOS:EOS_GRID:"' + \
        '{0:s}":MOD_Grid_BRDF:BRDF_Albedo_Parameters_Band{1:d}'.format(
        fname_a1, band)
    g = gdal.Open(fname)
    proj = g.GetProjection()
    geoT = g.GetGeoTransform()
    kernels = g.ReadAsArray()
    mask = np.all(kernels != 32767, axis=0)
    data = np.where(mask, kernels * 0.001, np.nan)
    fname = 'HDF4_EOS:EOS_GRID:' + \
        '"{0:s}":MOD_Grid_BRDF:BRDF_Albedo_Quality'.format(fname_a2)
    g = gdal.Open(fname)
    qa = g.ReadAsArray()
        #fname = 'HDF4_EOS:EOS_GRID:' + \
        #'"{0:s}":MOD_Grid_BRDF:BRDF_Albedo_Ancillary'.format(fname_a2)
    #g = gdal.Open(fname)
    #land = g.ReadAsArray()
    #land = np.bitwise_and(land, 112)

    mask = qa == 255 # crap pixel
    out = data*0.
    for i in xrange(3):
        out[i,:] = np.where(mask, np.nan, data[i,:])
    return out, proj, geoT

def find_granules(the_dir, tile, year, doy_start, doy_end):
    """
    Finds granules in a given directory, for all MCD43 datasets 
    between two dates. Returns two lists, one with the A1 and another
    with the A2 products.
    """
    fnames_a1 = glob.glob(os.path.join( the_dir, 
        "MCD43A1.A%d*.%s.*hdf" % (year, tile)))
    fnames_a1.sort()
    fnames_a2 = glob.glob(os.path.join( the_dir, 
        "MCD43A2.A%d*.%s.*hdf" % (year, tile)))
    fnames_a2.sort()
    a1_files = []
    a2_files = []
    for fich in fnames_a1:
        the_doy = int(fich.split("/")[-1].split(".")[1][-3:])
        if the_doy <= doy_end and the_doy >= doy_start:
            a1_files.append (fich)
            for fich2 in fnames_a2:
                the_doy2 = int(fich2.split("/")[-1].split(".")[1][-3:])
                if the_doy2 == the_doy:
                    a2_files.append(fich2)
    return a1_files, a2_files
    
def do_command_line():
    parser = argparse.ArgumentParser(description=__doc__)
    required = parser.add_argument_group('required arguments')
    parser.add_argument("data_folder", type=str,
                        help="Where the MCD43 files are")
    parser.add_argument("tile", type=str,
                        help="The MODIS tile")
    required.add_argument("-y", "--year", dest="year", action="store",
                        type=int, help="Year")
    required.add_argument("-s", "--start", dest="doy_start", action="store",
                        type=int, help="Starting DoY")
    required.add_argument("-e", "--end", dest="doy_end", action="store",
                        type=int, help="Ending DoY")
    required.add_argument("-o", "--out", dest="output_folder", action="store", 
                        type=str, help="Where the GeoTif will be saved")
    
    args = parser.parse_args()
    return (args.data_folder, args.tile, args.year, args.doy_start,
            args.doy_end, args.output_folder)


if __name__ == "__main__":

    data_dir, tile, year, doy_start, doy_end, output_dir = do_command_line()
    if not 1 <= doy_start <= 366:
        raise ValueError, "Starting date out of range: %d" % doy_start
    if not 1 <= doy_end <= 366:
        raise ValueError, "Ending date out of range: %d" % doy_start
    
    if doy_end <= doy_start:
        raise ValueError, "Wrong starting dates"
    
    for band in xrange(1,8):
        print "Doing band %d" % band
        a1_files, a2_files = find_granules(data_dir, tile,
            year, doy_start, doy_end)
            
        par1 = []
        par2 = []
        par3 = []
        for f1, f2 in zip(a1_files, a2_files):
            print "\t>>>Reading %s" % f1
            data, proj, geoT = read_mcd43 (band, f1, f2)
            par1.append ( data[0, :, :])
            par2.append ( data[1, :, :])
            par3.append ( data[2, :, :])
        P1 = np.nanmean( par1, axis=0 ).astype(np.float32)
        P2 = np.nanmean( par2, axis=0 ).astype(np.float32)
        P3 = np.nanmean( par3, axis=0 ).astype(np.float32)
        S1 = np.nanstd( par1, axis=0 ).astype(np.float32)
        S2 = np.nanstd( par2, axis=0 ).astype(np.float32)
        S3 = np.nanstd( par3, axis=0 ).astype(np.float32)
        Nsamples = np.sum(~np.isnan(par1), axis=0).astype(np.float32)
        drv = gdal.GetDriverByName("GTiff")
        dst_ds = drv.Create (os.path.join(output_dir, 
            "MCD43_average_%04d_%03d_%03d_b%d.tif" %(
            year, doy_start, doy_end, band)),
            2400, 2400, 7, gdal.GDT_Float32, ['COMPRESS=DEFLATE', 
            'BIGTIFF=YES', 'PREDICTOR=1','TILED=YES'])
        print "Saving to MCD43_average_%04d_%03d_%03d_b%d.tif" %(
            year, doy_start, doy_end, band)
        dst_ds.GetRasterBand(1).WriteArray(P1) # Mean isotropic
        dst_ds.GetRasterBand(2).WriteArray(P2) # Mean volumetric
        dst_ds.GetRasterBand(3).WriteArray(P3) # Mean geometric
        dst_ds.GetRasterBand(4).WriteArray(S1) # Std isotropic
        dst_ds.GetRasterBand(5).WriteArray(S2) # Std volumetric
        dst_ds.GetRasterBand(6).WriteArray(S3) # Std geometric
        dst_ds.GetRasterBand(7).WriteArray(Nsamples) # Number of samples
        dst_ds = None
        print "Done!"
    
