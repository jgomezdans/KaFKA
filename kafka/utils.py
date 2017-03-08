#!/usr/bin/env python
"""Some utility functions used by the main code base."""

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


import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import netCDF4
import datetime as dt
import os
import gdal


def spsolve2(a, b):
    a_lu = spl.splu(a.tocsc()) # LU decomposition for sparse a
    out = sp.lil_matrix((a.shape[1], b.shape[1]), dtype=np.float32)
    b_csc = b.tocsc()
    for j in xrange(b.shape[1]):
        bb = np.array(b_csc[j, :].todense()).squeeze()
        out[j,j] = a_lu.solve(bb)[j]
    return out.tocsr()


def matrix_squeeze (a_matrix, mask=None, n_params=1):
    """The matrix A is squeezed out of 0-filled submatrices, or rows/cols where
     a 1D mask indicates False. We return a squeezed version of the original
     matrix.
    Parameters
    ----------
    a_matrix: array
        An N,N array, or an N array. We assume that if no mask is given, we want
        to squeeze out all the zero locations
    mask: array
        An N boolean array, indicating where to squeeze the original array
    n_params: integer
        The number of parameters in the state vector. So for an identity o
        operator this will be 1, for the kernels 3, for TIP, 7, ...
    Returns
    -------
    The squeezed matrix.
    """
    if mask is None:
        # a needs to be a sparse matrix, otherwise find doesn't work!
        rows, columns, values = sp.find(a_matrix)
        # We can now squeeze easily using slicing of the original matrix
        a_matrix_squeezed = a_matrix[rows, :][:, columns]
    else:
        # Calculate the size of the output array from the non-zero mask elements
        if mask.ndim == 2: 
            mask = mask.ravel()
        n = mask.sum()
        a_matrix_squeezed = sp.csr_matrix((n, n))
        m = np.array([], dtype=np.bool)
        # the next if statement is there to cope with problems where the size of
        # the state has more than one parameter
        if n_params > 1:
            # We need to stack the masks in this case
            for i in xrange(n_params):
                m = np.r_[m, mask]
        else:
            # We don't stack the masks, as there's only one parameter
            m = mask
        # This is different for vector and matrix
        if a_matrix.ndim == 2:
            # Just subset by mask location in rows/cols
            a_matrix_squeezed = a_matrix[m, :][:, m]
        elif a_matrix.ndim == 1: # vector
            # Same, but just in one dimension
            a_matrix_squeezed = np.zeros(n_params*n)
            a_matrix_squeezed = a_matrix[m]
    return a_matrix_squeezed


def reconstruct_array(a_matrix, b_matrix, mask, n_params=1):
    """A function to fill in a squeezed array (a_matrix) with elements from a
    complete array (b_matrix). In effect, the elements of the b_matrix where the
    mask is True will be updated with the elements of a_matrix that correspond.
    The function works both on vectors and matrices, and they need to be
    ordered.
    Parameters
    -----------
    a_matrix: array
        The squeezed matrix with the updated elements
    b_matrix: array
        The full matrix that needs updating
    mask: array
        The location of the elements that need to be updated
    n_params: integer
        The number of parameters in the state
    Returns
    --------
    The updated `b_matrix`"""

    mask = mask.ravel()
    if a_matrix.ndim == 1:
        n = mask.shape[0] # big dimension
        n_good = mask.sum()
        ilocs = mask.nonzero()[0]
        for i in xrange(n_params):
            b_matrix[ilocs +i*n] = a_matrix[(i*n_good):((i+1)*n_good)]
    elif a_matrix.ndim == 2:
        n = mask.shape[0] # big dimension
        n_good = mask.sum()
        ilocs = mask.nonzero()[0]
        for i in xrange(n_params):
            ii = 0
            for j in xrange(n):
                if mask[j]:
                    b_matrix[j + i*n, i*n + ilocs] = a_matrix[ii,
                                                     (i*n_good):((i+1)*n_good)]
                    ii += 1
    return b_matrix


class OutputFile(object):
    """A netCDF4 class for saving output data. This class requires both the
    netCDF4 bindings, as well as the GDAL ones, as the class can take a GDAL
    projection to store it in the netCDF using CF-1.7. Lifted shamelessly from
    my eoldas code."""
    def __init__(self, fname, times=None, input_file=None,
                 basedate=dt.datetime(1970, 1, 1, 0, 0, 0),
                 x=None, y=None):
        """

        Parameters
        ----------
        fname : str
            A filename for the netCDF4 file.
        times : None
            Either None (default), or a list of datetime objects containing the
            dates of the data. The time axis of the data will be based on these.
            so you must prescribe the time before you save the data.
        input_file : str
            You can use a GDAL object (fully georeferenced) to copy the
            dimensions and projection.
        basedate : datetime
            The starting date, an arcane concept from netCDF.
        x : None or array
            A list of eastings if not passing an `input_file`.
        y : None or array
            A list of northings if not passing an `input_file`.
        """
        self.fname = fname
        self.basedate = basedate

        self.create_netcdf(times)

        if x is not None and y is not None:
            self.nc.createDimension('x', len(x))
            self.nc.createDimension('y', len(y))

            xo = self.nc.createVariable('x', 'f4', ('x'))
            xo.units = 'm'
            xo.standard_name = 'projection_x_coordinate'

            yo = self.nc.createVariable('y', 'f4', ('y'))
            yo.units = 'm'
            xo[:] = x
            yo[:] = y

        if input_file is not None:
            self._get_spatial_metadata(input_file)
            self.create_spatial_domain()

    def _get_spatial_metadata(self, geofile):
        """
        Gets (and sets!) the spatial metadata from a GDAL file
        Parameters
        ----------
        geofile : str
            The GDAL file. You might need to give a full path and what not...

        Returns
        -------
        Nuffink
        """
        g = gdal.Open(geofile)
        if g is None:
            raise IOError("File %s not readable by GDAL" % geofile)
        ny, nx = g.RasterYSize, g.RasterXSize
        geo_transform = g.GetGeoTransform()
        self.x = np.arange(nx) * geo_transform[1] + geo_transform[0]
        self.y = np.arange(ny) * geo_transform[5] + geo_transform[3]
        self.nx = nx
        self.ny = ny
        self.wkt = g.GetProjectionRef()

    def create_netcdf(self, times=None):
        """
        Creates our beloved netCDF file. Can also take an optional array of
        times.

        Parameters
        ----------
        times :

        Returns
        -------

        """
        self.nc = netCDF4.Dataset(self.fname, 'w', clobber=True)

        self.nc.createDimension('scalar', None)

        # create dimensions, variables and attributes:
        if times is None:
            self.nc.createDimension('time', None)
        else:
            self.nc.createDimension('time', len(times))
        timeo = self.nc.createVariable('time', 'f4', ('time'))
        timeo.units = 'days since 1858-11-17 00:00:00'
        timeo.standard_name = 'time'
        timeo.calendar = "Gregorian"
        if times is not None:
            timeo[:] = netCDF4.date2num(times, units=timeo.units,
                                        calendar=timeo.calendar)

    def create_spatial_domain(self):
        self.nc.createDimension('x', self.nx)
        self.nc.createDimension('y', self.ny)

        xo = self.nc.createVariable('x', 'f4', ('x'))

        xo.units = 'm'
        xo.standard_name = 'projection_x_coordinate'

        yo = self.nc.createVariable('y', 'f4', ('y'))
        yo.units = 'm'
        self.nc.Conventions = 'CF-1.7'
        # create container variable for CRS: x/y WKT
        try:
            crso = self.nc.createVariable('crs', 'i4')
            crso.grid_mapping_name = 'srs'
            crso.crs_wkt = self.wkt
        except AttributeError:
            print "Can't create a georeferenced netCDF file. Don't know why"
        xo[:] = self.x
        yo[:] = self.y


    def create_group(self, group):
        self.nc.createGroup(group)

    def create_variable(self, group, varname, vardata,
                        units, long_name, std_name, vartype='f4'):
        """
        Creates a variable in the file. The idea being that this is where data
        gets stored.
        MISSING STUFF:
        * Creating the variable without data (e.g. to append later)
        * Chunking!
        Parameters
        ----------
        group : str
            The netCDF group where the variable goes
        varname : str
            The variable name
        vardata : array
            The variable in a numpy array
        units : str
            The SI units (or more likely not)
        long_name : str
            The long variable name
        std_name : str
            The handy shorthand name
        vartype : str
            The variable type

        Returns
        -------

        """
        if vardata.ndim == 1:
            varo = self.nc.groups[group].createVariable(varname, vartype,
                                                        ('time'),
                                                        zlib=True,
                                                        chunksizes=[16],
                                                        fill_value=-9999)
            varo[:] = vardata
        elif vardata.ndim == 2:
            varo = self.nc.groups[group].createVariable(varname, vartype,
                                                        ('y', 'x'),
                                                        zlib=True,
                                                        chunksizes=[12, 12],
                                                        fill_value=-9999)
            varo.grid_mapping = 'crs'

            varo[:, :] = vardata

        elif vardata.ndim == 3:
            varo = self.nc.groups[group].createVariable(varname, vartype,
                                                        ('t', 'y', 'x'),
                                                        zlib=True,
                                                        chunksizes=[16, 12, 12],
                                                        fill_value=-9999)
            varo.grid_mapping = 'crs'
            varo[:, :, :] = vardata
        else:
            varo = self.nc.groups[group].createVariable(varname, vartype,
                                                        'scalar')
            varo[:] = vardata

        varo.units = units
        varo.scale_factor = 1.00
        varo.add_offset = 0.00
        varo.long_name = long_name
        varo.standard_name = std_name
        # varo.grid_mapping = 'crs'
        varo.set_auto_maskandscale(False)
        # varo[:,...] = vardata

    def __del__(self):
        self.nc.close()
