#!/usr/bin/env python
# ******************************************************************************
#  Name:     register.py
#  Purpose:  Perfrom image-image registration in frequency domain
#  Usage:             
#    python register.py 
#
#  Copyright (c) 2013, Mort Canty
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

import getopt
import os
import sys
import time

import numpy as np
import scipy.ndimage.interpolation as ndii
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly, GDT_Int16

from auxil.auxil import similarity

usage = '''
Usage:
------------------------------------------------
python %s [-h] [-b warpband] [-d "spatialDimensions"] reffname warpfname

Choose a reference image, the image to be warped and, optionally,
the band to be used for warping (default band 1) and the spatial subset
of the reference image.

The reference image should be smaller than the warp image
(i.e., the warp image should overlap the reference image completely)
and its upper left corner should be near that of the warp image:
----------------------
|   warp image
|
|  --------------------
|  |
|  |  reference image
|  |

The reference image (or spatial subset) should not contain zero data

The warped image (warpfile_warp) will be trimmed to the spatial
dimensions of the reference image.
------------------------------------------------'''


def main(img_ref, img_target, warpband=1, dims=None):

    gdal.AllRegister()

    print('------------REGISTER-------------')
    print(time.asctime())
    print('reference image: ' + img_ref)
    print('warp image: ' + img_target)
    print('warp band: %i' % warpband)

    start = time.time()

    path = os.path.dirname(os.path.abspath(img_target))
    basename2 = os.path.basename(img_target)
    root2, ext2 = os.path.splitext(basename2)
    outfn = os.path.join(path, root2 + '_warp' + ext2)
    inDataset1 = gdal.Open(img_ref, GA_ReadOnly)
    inDataset2 = gdal.Open(img_target, GA_ReadOnly)
    try:
        cols1 = inDataset1.RasterXSize
        rows1 = inDataset1.RasterYSize
        cols2 = inDataset2.RasterXSize
        rows2 = inDataset2.RasterYSize
        bands2 = inDataset2.RasterCount
    except Exception as e:
        print('Error %s  --Image could not be read in' % e)
        sys.exit(1)
    if dims is None:
        x0 = 0
        y0 = 0
    else:
        x0, y0, cols1, rows1 = dims

    band = inDataset1.GetRasterBand(warpband)
    refband = band.ReadAsArray(x0, y0, cols1, rows1).astype(np.float32)
    band = inDataset2.GetRasterBand(warpband)
    warpband = band.ReadAsArray(x0, y0, cols1, rows1).astype(np.float32)

    #  similarity transform parameters for reference band number
    scale, angle, shift = similarity(refband, warpband)

    driver = inDataset2.GetDriver()
    outDataset = driver.Create(outfn, cols1, rows1, bands2, GDT_Int16)
    projection = inDataset1.GetProjection()
    geotransform = inDataset1.GetGeoTransform()
    if geotransform is not None:
        gt = list(geotransform)
        gt[0] = gt[0] + x0 * gt[1]
        gt[3] = gt[3] + y0 * gt[5]
        outDataset.SetGeoTransform(tuple(gt))
    if projection is not None:
        outDataset.SetProjection(projection)

        #  warp
    for k in range(bands2):
        inband = inDataset2.GetRasterBand(k + 1)
        outBand = outDataset.GetRasterBand(k + 1)
        bn1 = inband.ReadAsArray(0, 0, cols2, rows2).astype(np.float32)
        bn2 = ndii.zoom(bn1, 1.0 / scale)
        bn2 = ndii.rotate(bn2, angle)
        bn2 = ndii.shift(bn2, shift)
        outBand.WriteArray(bn2[y0:y0 + rows1, x0:x0 + cols1])
        outBand.FlushCache()

    del inDataset1
    del inDataset2
    del outDataset
    print('Warped image written to: %s' % outfn)
    print('elapsed time: %s' % str(time.time() - start))

    return outfn

if __name__ == '__main__':
    options, args = getopt.getopt(sys.argv[1:], 'hb:d:')
    warpband = 1
    dims = None
    for option, value in options:
        if option == '-h':
            print(usage)
            sys.exit()
        elif option == '-b':
            warpband = eval(value)
        elif option == '-d':
            dims = eval(value)
    if len(args) != 2:
        print('Incorrect number of arguments')
        print(usage)
        sys.exit(1)

    fn1 = args[0]  # reference
    fn2 = args[1]  # warp

    main(fn1, fn2, warpband=warpband, dims=dims)
