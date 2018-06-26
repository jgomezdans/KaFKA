import numpy as np
import gdal
import osr
"""
I need to put some utils in here. Seems like the most obvious place...
"""


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
            # find Y offset
            this_Y = Y * block_size[1]
            yield this_X, this_Y, nx_valid, ny_valid, chunk_no


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

def raster_extent_feature(raster):
    """Gets a geometry with the extent of the raster file in WGS84 coordinates"""
    raster = gdal.Open(raster)
    wgs84 = osr.SpatialReference()
    wgs84.ImportFromEPSG(4326)
    src = osr.SpatialReference()
    src.ImportFromWkt(raster.GetProjection())
    coord_transform = osr.CoordinateTransformation(src, wgs84)
    
    transform = raster.GetGeoTransform()
    pixelWidth = transform[1]
    pixelHeight = transform[5]
    cols = raster.RasterXSize
    rows = raster.RasterYSize

    xLeft = transform[0]
    yTop = transform[3]
    xRight = xLeft + cols*pixelWidth
    yBottom = yTop + rows*pixelHeight

    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(xLeft, yTop)
    ring.AddPoint(xLeft, yBottom)
    ring.AddPoint(xRight, yTop)
    ring.AddPoint(xRight, yBottom)
    ring.AddPoint(xLeft, yTop)
    raster_geometry = ogr.Geometry(ogr.wkbPolygon)
    raster_geometry.AddGeometry(ring)
    raster_geometry.Transform(coord_transform)
    return raster_geometry
    

def find_overlap_raster_feature(raster, feature):
    # First, reproject feature to lat/long
    wgs84 = osr.SpatialReference()
    wgs84.ImportFromEPSG(4326)
    src = osr.SpatialReference()
    src.ImportFromWkt(feature.GetSpatialReference().ExportToWkt())
    coord_transform = osr.CoordinateTransformation(src, wgs84)
    feature.Transform(coord_transform)
    # Now get extent of the raster file in lat/long
    raster_feature = raster_extent_feature(raster)
    return feature.Intersects(raster_feature)
    
    
