#!/usr/bin/env python3
# ******************************************************************************
#  Name:     radcal.py
#  Purpose:  Automatic radiometric normalization using IR-MAD invariants.
#
#  Given an IR-MAD output (MAD variates + chi-square band) plus the original
#  reference/target images, this module:
#    1. Selects no-change pixels via the chi-square 'no-change probability'.
#    2. Fits a per-band orthogonal regression target -> reference on those.
#    3. Applies the linear transform a + b*target to produce a normalized image.
#
#  Original implementation: Mort Canty, 2011. Refactored for numerical
#  stability and to match the rest of the package style.
#
#  License: GPLv2+
# ******************************************************************************

import getopt
import os
import sys
import time

import numpy as np
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly
from scipy import stats

from arrnorm.auxil.auxil import orthoregress

usage = '''
Usage:
--------------------------------------------------------
python radcal.py [-p "bandPositions"] [-d "spatialDimensions"]
                 [-t no-change prob threshold] imadFile [fullSceneFile]
--------------------------------------------------------
'''


# Output range per GDAL integer type. Used to clip the linear-regression
# output before writing to disk, so negative or saturated values are
# safely capped instead of silently wrapping around.
_GDT_RANGES = {
    gdal.GDT_Byte:   (0, 255),
    gdal.GDT_UInt16: (0, 65535),
    gdal.GDT_Int16:  (-32768, 32767),
    gdal.GDT_UInt32: (0, 4294967295),
    gdal.GDT_Int32:  (-2147483648, 2147483647),
}


def _clip_for_dtype(arr, gdal_dtype):
    """Clip array to the valid output range for the given GDAL dtype.

    Float dtypes are returned unchanged. Without this, integer outputs
    silently wrap on overflow / negative values (the previous behaviour).
    """
    rng = _GDT_RANGES.get(gdal_dtype)
    if rng is None:
        return arr
    return np.clip(arr, rng[0], rng[1])


def main(img_imad, ncpThresh=0.95, pos=None, dims=None, img_target=None,
         graphics=False, out_dtype=None):

    if img_target is not None:
        path = os.path.dirname(img_target)
        basename = os.path.basename(img_target)
        root, ext = os.path.splitext(basename)
        fsoutfn = os.path.join(path, root + '_norm_all' + ext)

    path = os.path.dirname(img_imad)
    basename = os.path.basename(img_imad)
    root, ext = os.path.splitext(basename)
    b = root.find('(')
    err_idx = root.find(')')
    referenceroot, targetbasename = root[b + 1:err_idx].split('&')
    referencefn = os.path.join(path, referenceroot + ext)
    targetfn = os.path.join(path, targetbasename)
    targetroot, targetext = os.path.splitext(targetbasename)
    outfn = os.path.join(path, targetroot + '_norm' + targetext)

    imadDataset = gdal.Open(img_imad, GA_ReadOnly)
    if imadDataset is None:
        sys.stderr.write(f'Error: could not open iMAD file: {img_imad}\n')
        sys.exit(1)
    imadbands = imadDataset.RasterCount
    cols = imadDataset.RasterXSize
    rows = imadDataset.RasterYSize

    referenceDataset = gdal.Open(referencefn, GA_ReadOnly)
    targetDataset = gdal.Open(targetfn, GA_ReadOnly)
    if referenceDataset is None or targetDataset is None:
        sys.stderr.write('Error: could not open reference/target image.\n')
        sys.exit(1)

    if pos is None:
        pos = list(range(1, referenceDataset.RasterCount + 1))
    if dims is None:
        x0 = y0 = 0
    else:
        x0, y0, cols, rows = dims

    # The last iMad band is the chi-square statistic over the MAD variates.
    # Under the null hypothesis (no change), it follows chi^2 with
    # (imadbands - 1) degrees of freedom — the # of MAD variates.
    # NCP = P(X >= chisqr) = sf(chisqr), which is numerically far more
    # accurate than 1 - cdf() in the relevant upper tail.
    chisqr = imadDataset.GetRasterBand(imadbands).ReadAsArray(0, 0, cols, rows).ravel()
    ncp = stats.chi2.sf(chisqr, imadbands - 1)
    idx = np.where(ncp > ncpThresh)
    print(time.asctime())
    print(f'reference: {referencefn}')
    print(f'target   : {targetfn}')
    print(f'no-change probability threshold: {ncpThresh}')
    print(f'no-change pixels: {len(idx[0])}')

    if len(idx[0]) < 2:
        sys.stderr.write(
            f"Error: only {len(idx[0])} no-change pixels selected "
            f"(threshold={ncpThresh}). Lower -t to keep more pixels.\n")
        sys.exit(1)

    start = time.time()
    driver = targetDataset.GetDriver()
    outDataset = driver.Create(outfn, cols, rows, len(pos), out_dtype)
    projection = imadDataset.GetProjection()
    geotransform = imadDataset.GetGeoTransform()
    if geotransform is not None:
        outDataset.SetGeoTransform(geotransform)
    if projection is not None:
        outDataset.SetProjection(projection)

    aa = []
    bb = []
    if graphics:
        try:
            import matplotlib.pyplot as plt
            plt.figure(1, (9, 6))
        except ImportError:
            graphics = False

    bands = len(pos)
    for j, k in enumerate(pos, start=1):
        x = referenceDataset.GetRasterBand(k).ReadAsArray(x0, y0, cols, rows).astype(np.float64).ravel()
        y = targetDataset.GetRasterBand(k).ReadAsArray(x0, y0, cols, rows).astype(np.float64).ravel()
        b_slope, a_intercept, R = orthoregress(y[idx], x[idx])
        print(f'band: {k}  slope: {b_slope:.6f}  intercept: {a_intercept:.6f}  correlation: {R:.6f}')
        my = float(np.max(y[idx]))
        if (j < 7) and graphics:
            plt.subplot(2, 3, j)
            plt.plot(y[idx], x[idx], '.')
            plt.plot([0, my], [a_intercept, a_intercept + b_slope * my])
            plt.title(f'Band {k}')
            if ((j < 4) and (bands < 4)) or j > 3:
                plt.xlabel('Target')
            if (j == 1) or (j == 4):
                plt.ylabel('Reference')
        aa.append(a_intercept)
        bb.append(b_slope)
        outBand = outDataset.GetRasterBand(j)
        normalized = a_intercept + b_slope * y
        normalized = _clip_for_dtype(normalized, out_dtype)
        outBand.WriteArray(normalized.reshape(rows, cols), 0, 0)
        outBand.FlushCache()

    if graphics:
        plt.show()
        plt.close()
    referenceDataset = None
    targetDataset = None
    outDataset = None
    print(f'result written to: {outfn}')

    if img_target is not None:
        print(f'normalizing {img_target}...')
        fsDataset = gdal.Open(img_target, GA_ReadOnly)
        if fsDataset is None:
            sys.stderr.write(f'Error: full-scene file could not be opened: {img_target}\n')
            sys.exit(1)
        fcols = fsDataset.RasterXSize
        frows = fsDataset.RasterYSize
        driver = fsDataset.GetDriver()
        outDataset = driver.Create(fsoutfn, fcols, frows, len(pos), out_dtype)
        projection = fsDataset.GetProjection()
        geotransform = fsDataset.GetGeoTransform()
        if geotransform is not None:
            outDataset.SetGeoTransform(geotransform)
        if projection is not None:
            outDataset.SetProjection(projection)
        for j, k in enumerate(pos, start=1):
            inBand = fsDataset.GetRasterBand(k)
            outBand = outDataset.GetRasterBand(j)
            for i in range(frows):
                y = inBand.ReadAsArray(0, i, fcols, 1).astype(np.float64)
                normalized = aa[j - 1] + bb[j - 1] * y
                normalized = _clip_for_dtype(normalized, out_dtype)
                outBand.WriteArray(normalized, 0, i)
            outBand.FlushCache()
        outDataset = None
        fsDataset = None
        print(f'full result written to: {fsoutfn}')
        return fsoutfn

    print(f'elapsed time: {time.time() - start:.2f}s')
    return outfn


if __name__ == '__main__':
    options, args = getopt.getopt(sys.argv[1:], 'hnp:d:t:')
    pos = None
    dims = None
    ncpThresh = 0.95
    fsfn = None
    graphics = True
    for option, value in options:
        if option == '-h':
            print(usage)
            sys.exit()
        elif option == '-n':
            graphics = False
        elif option == '-p':
            pos = eval(value)
        elif option == '-d':
            dims = eval(value)
        elif option == '-t':
            ncpThresh = float(value)
    if (len(args) != 1) and (len(args) != 2):
        print('Incorrect number of arguments')
        print(usage)
        sys.exit(1)
    imadfn = args[0]
    if len(args) == 2:
        fsfn = args[1]

    main(imadfn, ncpThresh, pos, dims, fsfn, graphics=graphics)
