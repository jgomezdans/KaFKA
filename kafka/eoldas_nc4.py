import cPickle
import netCDF4 
import numpy as np

import numpy as np
import datetime as dt
import os
import gdal
import netCDF4

#from best_chunk import chunk_shape_3D


def store_sparse_mat(m, name, store='store.h5'):
    msg = "This code only works for csr matrices"
    assert(m.__class__ == sparse.csr.csr_matrix), msg
    with tb.openFile(store,'a') as f:
        for par in ('data', 'indices', 'indptr', 'shape'):
            full_name = '%s_%s' % (name, par)
            try:
                n = getattr(f.root, full_name)
                n._f_remove()
            except AttributeError:
                pass

            arr = array(getattr(m, par))
            atom = tb.Atom.from_dtype(arr.dtype)
            ds = f.createCArray(f.root, full_name, atom, arr.shape)
            ds[:] = arr

def load_sparse_mat(name, store='store.h5'):
    with tb.openFile(store) as f:
        pars = []
        for par in ('data', 'indices', 'indptr', 'shape'):
            pars.append(getattr(f.root, '%s_%s' % (name, par)).read())
    m = sparse.csr_matrix(tuple(pars[:3]), shape=pars[3])
    return m

class OutputFile ( object ):
    
    def __init__ ( self, fname, times=None, input_file=None,            
                        basedate=dt.datetime(1970,1,1,0,0,0),
                        x=None, y=None):
        self.fname = fname
        self.basedate = basedate

        self.create_netcdf ( times )
        if x is not None and y is not None:
            self.nc.createDimension( 'x', len(x) )
            self.nc.createDimension( 'y', len(y) )

            xo = self.nc.createVariable('x','f4',('x'))
            xo.units = 'm'
            xo.standard_name = 'projection_x_coordinate'

            yo = self.nc.createVariable('y','f4',('y'))
            yo.units = 'm'            
            xo[:] = x
            yo[:] = y
            
        if input_file is not None:
            self._get_spatial_metadata ( input_file )
            self.create_spatial_domain( )
        
    def _get_spatial_metadata ( self, geofile ):
        g = gdal.Open ( geofile )
        if g is None:
            raise IOError ("File %s not readable by GDAL" % geofile )
        ny, nx = g.RasterYSize, g.RasterXSize
        geo_transform = g.GetGeoTransform ()
        self.x = np.arange ( nx )*geo_transform[1] + geo_transform[0]
        self.y = np.arange ( ny )*geo_transform[5] + geo_transform[3]
        self.nx = nx
        self.ny = ny
        self.wkt = g.GetProjectionRef()

    def create_netcdf ( self, times=None ):
        
        self.nc = netCDF4.Dataset( self.fname, 'w', clobber=True )

        self.nc.createDimension ( 'scalar', None )

        # create dimensions, variables and attributes:
        if times is None:
            self.nc.createDimension( 'time', None )
        else:
            self.nc.createDimension( 'time', len ( times ) )
        timeo = self.nc.createVariable( 'time', 'f4', ('time') )
        timeo.units = 'days since 1858-11-17 00:00:00'
        timeo.standard_name = 'time'
        timeo.calendar = "Gregorian"
        if times is not None:
            timeo[:] = netCDF4.date2num ( times, units=timeo.units, 
                                         calendar=timeo.calendar )
        
    def create_spatial_domain ( self ):
        self.nc.createDimension( 'x', self.nx )
        self.nc.createDimension( 'y', self.ny )

        xo = self.nc.createVariable('x','f4',('x'))
        xo.units = 'm'
        xo.standard_name = 'projection_x_coordinate'

        yo = self.nc.createVariable('y','f4',('y'))
        yo.units = 'm'

        # create container variable for CRS: x/y WKT
        crso = self.nc.createVariable('crs','i4')
        crso.grid_mapping_name ( srs )
        crso.crs_wkt ( self.wkt )
        xo[:] = self.x
        yo[:] = self.y
        self.nc.Conventions='CF-1.7'

    def create_group ( self, group ):
        self.nc.createGroup ( group )

    def create_variable ( self, group, varname, vardata,
            units, long_name, std_name, vartype='f4' ):
        if vardata.ndim == 1:
            varo = self.nc.groups[group].createVariable(varname, vartype,  ('time'), 
                zlib=True,chunksizes=[16],fill_value=-9999)
            varo[:] = vardata
        elif vardata.ndim == 2:
            varo = self.nc.groups[group].createVariable(varname, vartype,  ('y', 'x'), 
                zlib=True,chunksizes=[12, 12],fill_value=-9999)
            varo.grid_mapping = 'crs'
        
            varo[:,:] = vardata

        elif vardata.ndim == 3:
            varo = self.nc.groups[group].createVariable(varname, vartype,  ( 't','y', 'x'), 
                zlib=True,chunksizes=[16, 12, 12],fill_value=-9999)
            varo.grid_mapping = 'crs'
            varo[:,:, :] = vardata
        else:
            varo = self.nc.groups[group].createVariable(varname, vartype,  'scalar')
            varo[:] = vardata
            

        varo.units = units
        varo.scale_factor = 1.00
        varo.add_offset = 0.00
        varo.long_name = long_name
        varo.standard_name = std_name
        # varo.grid_mapping = 'crs'
        varo.set_auto_maskandscale(False)
        #varo[:,...] = vardata
    
    def __del__ ( self ):
        self.nc.close()


def pkl_to_nc4 ( pk_file, nc_output ):

        with open ( pk_file, 'r' ) as fp:
            f = cPickle.load ( fp )
            nt=len(f['real_map']['lai'] )
            times = [dt.datetime ( 2001, 1, 1) + i*dt.timedelta(days=1) \
                        for i in xrange(nt) ]
            nc = OutputFile ( nc_output, times=times)
            for group in f.iterkeys():
                if group == "post_sigma" or group == "post_cov":
                    continue
                nc.create_group ( group )
                for field in f[group].iterkeys():
                    nc.create_variable ( group, field, f[group][field], "N/A", "",field )
        return nc
if __name__ == "__main__":
    nc =pkl_to_nc4 ( "/home/ucfajlg/python/da_esa_wkshp/eoldas_retval_20150905_194312_cubil.pkl", "/tmp/testme.nc" )
    print nc