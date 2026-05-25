#!/usr/bin/env python3
# ******************************************************************************
#  Name:     register.py
#  Purpose:  Image-image registration in the frequency domain.
#
#  Estimates a similarity transform (scale + rotation + translation) between
#  one band of the reference and target images using log-polar / Fourier
#  cross-correlation (Reddy & Chatterji), then applies it to every band of
#  the target. When chunksize is set, the estimation is repeated on a grid
#  of blocks and the warped blocks are mosaicked back together.
#
#  Original: M. Canty, 2013. Refactored: clearer variable names, mosaic via
#  the gdal.Warp Python API instead of subprocess.
#
#  License: GPLv2+
# ******************************************************************************

import ast
import getopt
import os
import sys
import time

import numpy as np
import scipy.ndimage as ndii
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly

from core.auxil.auxil import similarity

usage = '''
Usage:
------------------------------------------------
python register.py [-h] [-b warpband] [-d "spatialDimensions"]
                   reffname warpfname
------------------------------------------------'''


def _chunks(seq, n):
    """Split a sequence into consecutive chunks of size n (last may be smaller)."""
    n = max(1, n)
    return [seq[i:i + n] for i in range(0, len(seq), n)]


def main(img_ref, img_target, warpband=2, chunksize=None):
    gdal.AllRegister()

    print('------------REGISTER-------------')
    print(time.asctime())
    print(f'reference image: {img_ref}')
    print(f'warp image: {img_target}')
    print(f'warp band: {warpband}')

    start = time.time()

    path = os.path.dirname(os.path.abspath(img_target))
    basename2 = os.path.basename(img_target)
    root2, ext2 = os.path.splitext(basename2)
    outfn = os.path.join(path, root2 + '_warp' + ext2)

    inDataset1 = gdal.Open(img_ref, GA_ReadOnly)
    inDataset2 = gdal.Open(img_target, GA_ReadOnly)
    if inDataset1 is None or inDataset2 is None:
        sys.stderr.write('Error: input image(s) could not be opened.\n')
        sys.exit(1)

    max_band = min(inDataset1.RasterCount, inDataset2.RasterCount)
    if warpband > max_band:
        print(f'warpband {warpband} exceeds available bands ({max_band}); falling back to band 1')
        warpband = 1

    ref_band = inDataset1.GetRasterBand(warpband)
    tgt_band = inDataset2.GetRasterBand(warpband)

    cols1 = inDataset1.RasterXSize
    rows1 = inDataset1.RasterYSize
    cols2 = inDataset2.RasterXSize
    rows2 = inDataset2.RasterYSize
    bands2 = inDataset2.RasterCount

    if chunksize is not None:
        # Adjust block size so every block has a similar shape — np.ceil gives
        # us a divisor that comes close to `chunksize` without leaving tiny
        # remainder strips on the edges.
        x_chunk_size = int(np.ceil(cols2 / round(cols2 / chunksize)))
        y_chunk_size = int(np.ceil(rows2 / round(rows2 / chunksize)))
    else:
        x_chunk_size = cols1
        y_chunk_size = rows1

    x_chunks = _chunks(list(range(cols2)), x_chunk_size)
    y_chunks = _chunks(list(range(rows2)), y_chunk_size)

    blocks_files = []
    for y_idx_block, y_block in enumerate(y_chunks):
        for x_idx_block, x_block in enumerate(x_chunks):
            x0 = x_block[0]
            y0 = y_block[0]
            cols_blk = len(x_block)
            rows_blk = len(y_block)

            block_filename = os.path.join(
                path,
                f'{root2}_warp_block_x{x_idx_block}y{y_idx_block}{ext2}')

            # Estimate similarity transform on this block
            ref_tile = ref_band.ReadAsArray(x0, y0, cols_blk, rows_blk).astype(np.float32)
            warp_tile = tgt_band.ReadAsArray(x0, y0, cols_blk, rows_blk).astype(np.float32)
            scale, angle, shift = similarity(ref_tile, warp_tile)

            driver = inDataset2.GetDriver()
            outDataset = driver.Create(block_filename, cols_blk, rows_blk,
                                       bands2, tgt_band.DataType)
            projection = inDataset1.GetProjection()
            geotransform = inDataset1.GetGeoTransform()
            if geotransform is not None:
                gt = list(geotransform)
                gt[0] = gt[0] + x0 * gt[1]
                gt[3] = gt[3] + y0 * gt[5]
                outDataset.SetGeoTransform(tuple(gt))
            if projection is not None:
                outDataset.SetProjection(projection)

            # Apply the translation to every band (zoom/rotate intentionally
            # disabled — for Landsat-scale shifts they introduce more
            # interpolation noise than they remove)
            for k in range(bands2):
                in_band_full = inDataset2.GetRasterBand(k + 1)
                out_band_blk = outDataset.GetRasterBand(k + 1)
                bn1 = in_band_full.ReadAsArray(0, 0, cols2, rows2).astype(np.float32)
                shifted = ndii.shift(bn1, shift)
                out_band_blk.WriteArray(shifted[y0:y0 + rows_blk, x0:x0 + cols_blk])
                out_band_blk.FlushCache()

            blocks_files.append(block_filename)
            outDataset = None

    # Mosaic blocks via the in-process gdal.Warp API rather than spawning
    # an external `gdalwarp` subprocess — same result, no PATH/version
    # surprises, and we get the error message back as a Python exception.
    warp_result = gdal.Warp(outfn, blocks_files, format='GTiff')
    if warp_result is None:
        sys.stderr.write('Error: gdal.Warp failed to mosaic block files.\n')
        sys.exit(1)
    warp_result = None
    print('mosaic created successfully')

    # delete the temporary block files
    for chunk_file in blocks_files:
        try:
            os.remove(chunk_file)
        except OSError:
            pass

    inDataset1 = None
    inDataset2 = None
    print(f'Warped image written to: {outfn}')
    print(f'elapsed time: {time.time() - start:.2f}s')

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
            warpband = int(value)
        elif option == '-d':
            dims = ast.literal_eval(value)
    if len(args) != 2:
        print('Incorrect number of arguments')
        print(usage)
        sys.exit(1)

    main(args[0], args[1], warpband=warpband)
