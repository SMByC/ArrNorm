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
from subprocess import call
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly, GDT_UInt16

from arrnorm.auxil.auxil import similarity

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


def chunks(l, n):
    """Split a list into evenly sized chunks

    :param l: list to chunk
    :type l: list
    :param n: n sizes to chunk
    :type n: int
    :rtype: list
    """
    n = max(1, n)
    return [l[i:i + n] for i in range(0, len(l), n)]


def main(img_ref, img_target, warpband=2, chunksize=None):

    gdal.AllRegister()

    print('------------REGISTER-------------')
    print(time.asctime())
    print('reference image: ' + img_ref)
    print('warp image: ' + img_target)
    print('warp band: {}'.format(warpband))

    start = time.time()

    path = os.path.dirname(os.path.abspath(img_target))
    basename2 = os.path.basename(img_target)
    root2, ext2 = os.path.splitext(basename2)
    outfn = os.path.join(path, root2 + '_warp' + ext2)
    # reference
    inDataset1 = gdal.Open(img_ref, GA_ReadOnly)
    band_ref = inDataset1.GetRasterBand(warpband)
    # target
    inDataset2 = gdal.Open(img_target, GA_ReadOnly)
    band_targ = inDataset2.GetRasterBand(warpband)

    try:
        cols1 = inDataset1.RasterXSize
        rows1 = inDataset1.RasterYSize
        cols2 = inDataset2.RasterXSize
        rows2 = inDataset2.RasterYSize
        bands2 = inDataset2.RasterCount
    except Exception as err:
        print('Error {}  --Image could not be read in'.format(err))
        sys.exit(1)

    if chunksize is not None:
        # adjusted the chunks size
        x_chunk_size = int(np.ceil(cols2 / round(cols2 / chunksize)))
        y_chunk_size = int(np.ceil(rows2 / round(rows2 / chunksize)))
    else:
        x_chunk_size = cols1
        y_chunk_size = rows1

    x_chunks = chunks(list(range(cols2)), x_chunk_size)  # divide x in blocks with size 250
    y_chunks = chunks(list(range(rows2)), y_chunk_size)  # divide y in blocks with size 250

    blocks_files = []
    for y_idx_block in range(len(y_chunks)):
        for x_idx_block in range(len(x_chunks)):
            x0 = x_chunks[x_idx_block][0]
            y0 = y_chunks[y_idx_block][0]

            cols1 = len(x_chunks[x_idx_block])
            rows1 = len(y_chunks[y_idx_block])

            block_filename = os.path.join(path, root2 + '_warp_block_x' + str(x_idx_block)
                                          +'y' + str(y_idx_block) + ext2)

            ## make register in block
            refband = band_ref.ReadAsArray(x0, y0, cols1, rows1).astype(np.float32)
            warpband = band_targ.ReadAsArray(x0, y0, cols1, rows1).astype(np.float32)

            #  similarity transform parameters for reference band number
            scale, angle, shift = similarity(refband, warpband)

            driver = inDataset2.GetDriver()
            outDataset = driver.Create(block_filename, cols1, rows1, bands2, GDT_UInt16)
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
                #bn2 = ndii.zoom(bn1, 1.0 / scale)  # TODO
                #bn2 = ndii.rotate(bn2, angle)  # TODO
                bn2 = ndii.shift(bn1, shift)
                outBand.WriteArray(bn2[y0:y0 + rows1, x0:x0 + cols1])
                outBand.FlushCache()

            blocks_files.append(block_filename)
            del outDataset

    return_code = call(["gdalwarp"] + blocks_files + [outfn])  # [outfn, '-srcnodata', '0', '-dstnodata', '255']

    if return_code == 0:  # successfully
        print('mosaic created successfully')

    # delete the temporal chunk files
    for chunk_file in blocks_files:
        os.remove(chunk_file)

    del inDataset1
    del inDataset2
    print('Warped image written to: {}'.format(outfn))
    print('elapsed time: {}'.format(time.time() - start))

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
