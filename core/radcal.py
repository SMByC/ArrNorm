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

import ast
import getopt
import os
import sys
import time

import numpy as np
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly
from scipy import stats

from core.auxil.auxil import orthoregress

try:
    import matplotlib
    # 'Agg' is a non-interactive backend — required for safe use from
    # multiprocessing workers and for headless servers. Must be set before
    # importing pyplot.
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

usage = '''
Usage:
--------------------------------------------------------
python radcal.py [-p "bandPositions"] [-d "spatialDimensions"]
                 [--ncp-threshold no-change prob threshold] imadFile [fullSceneFile]
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


def main(img_imad, ncp_threshold=0.95, pos=None, dims=None, img_target=None,
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
    idx = np.where(ncp > ncp_threshold)
    print(time.asctime())
    print(f'reference: {referencefn}')
    print(f'target   : {targetfn}')
    print(f'no-change probability threshold: {ncp_threshold}')
    print(f'no-change pixels: {len(idx[0])}')

    if len(idx[0]) < 2:
        sys.stderr.write(
            f"Error: only {len(idx[0])} no-change pixels selected "
            f"(threshold={ncp_threshold}). Lower -t to keep more pixels.\n")
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
    if graphics and not _MPL_AVAILABLE:
        print('Warning: matplotlib not available — graphics output disabled.')
        graphics = False

    bands = len(pos)
    fig = None
    plot_nrows = plot_ncols = 0
    if graphics:
        # Lay out band scatters in a grid of at most 3 cols × 2 rows = 6 panels.
        plot_ncols = min(bands, 3)
        plot_nrows = 2 if bands > 3 else 1
        n_total = cols * rows
        n_nochange = len(idx[0])
        pct_nochange = 100.0 * n_nochange / n_total if n_total else 0.0
        fig, axes = plt.subplots(
            nrows=plot_nrows,
            ncols=plot_ncols,
            figsize=(4.2 * plot_ncols, 4.2 * plot_nrows),
            squeeze=False,
        )
        fig.suptitle(
            f'RadCal: per-band orthogonal regression on IR-MAD no-change pixels\n'
            f'target: {os.path.basename(targetfn)}   '
            f'reference: {os.path.basename(referencefn)}\n'
            f'no-change pixels: {n_nochange:,} / {n_total:,} '
            f'({pct_nochange:.2f}%)   '
            f'NCP threshold: {ncp_threshold}',
            fontsize=10,
        )

    for j, k in enumerate(pos, start=1):
        x = referenceDataset.GetRasterBand(k).ReadAsArray(x0, y0, cols, rows).astype(np.float64).ravel()
        y = targetDataset.GetRasterBand(k).ReadAsArray(x0, y0, cols, rows).astype(np.float64).ravel()
        b_slope, a_intercept, R = orthoregress(y[idx], x[idx])
        print(f'band: {k}  slope: {b_slope:.6f}  intercept: {a_intercept:.6f}  correlation: {R:.6f}')
        if graphics and j <= 6:
            row, col = divmod(j - 1, 3)
            ax = axes[row][col]

            xt = y[idx]          # target values at no-change pixels (x-axis)
            yr = x[idx]          # reference values at no-change pixels (y-axis)

            # Independent percentile-based axis limits — each axis is framed
            # tightly around its own data. Using one shared range for both
            # axes would push the data cluster into a corner whenever the
            # radiometric drift is large (e.g. target ~50, reference ~150).
            x_lo = float(np.percentile(xt, 1))
            x_hi = float(np.percentile(xt, 99))
            y_lo = float(np.percentile(yr, 1))
            y_hi = float(np.percentile(yr, 99))
            # Add 3% padding so points/lines aren't flush against the frame
            pad_x = 0.03 * (x_hi - x_lo) if x_hi > x_lo else 1.0
            pad_y = 0.03 * (y_hi - y_lo) if y_hi > y_lo else 1.0
            x_lo, x_hi = x_lo - pad_x, x_hi + pad_x
            y_lo, y_hi = y_lo - pad_y, y_hi + pad_y

            # Vertical-residual RMSE of the fit — informative even though
            # orthoregress minimizes perpendicular distance, because it's the
            # quantity actually applied to the target band on output.
            residuals = yr - (a_intercept + b_slope * xt)
            rmse = float(np.sqrt(np.mean(residuals ** 2)))

            # 1:1 reference line: only draw the segment of y=x that actually
            # falls inside the visible (x_lo..x_hi, y_lo..y_hi) window. If
            # there is no overlap (data is entirely above or below the
            # diagonal) the line is simply omitted.
            one_lo = max(x_lo, y_lo)
            one_hi = min(x_hi, y_hi)
            draw_one_to_one = one_hi > one_lo

            # Draw order: 1:1 reference behind, scatter mid, fit line on top.
            if draw_one_to_one:
                ax.plot([one_lo, one_hi], [one_lo, one_hi],
                        color='0.6', linestyle=':', lw=0.8, alpha=0.7,
                        zorder=1, label='1:1')
            ax.scatter(xt, yr, s=1, alpha=0.25,
                       color='steelblue', rasterized=True, zorder=2)
            line_x = np.array([x_lo, x_hi])
            ax.plot(line_x, a_intercept + b_slope * line_x,
                    color='crimson', lw=1.6, zorder=3,
                    label=f'fit: y = {a_intercept:.2f} + {b_slope:.3f}·x')

            ax.set_title(f'Band {k}   R²={R ** 2:.3f}   RMSE={rmse:.2f}',
                         fontsize=10)
            ax.set_xlabel('Target')
            ax.set_ylabel('Reference')
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(y_lo, y_hi)
            # Note: no aspect='equal' — equal aspect with independent axes
            # would distort the visible plot region; we let matplotlib choose
            # the aspect that fills each subplot.
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left', fontsize=8, framealpha=0.85)
        aa.append(a_intercept)
        bb.append(b_slope)
        outBand = outDataset.GetRasterBand(j)
        normalized = a_intercept + b_slope * y
        normalized = _clip_for_dtype(normalized, out_dtype)
        outBand.WriteArray(normalized.reshape(rows, cols), 0, 0)
        outBand.FlushCache()

    if graphics and fig is not None:
        # Hide unused axes when bands < grid capacity (e.g. 4 bands → 2×3 = 6 slots, 2 empty)
        for idx_ax in range(bands, plot_nrows * plot_ncols):
            r, c = divmod(idx_ax, plot_ncols)
            axes[r][c].set_visible(False)
        # Reserve some headroom for the multi-line suptitle
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        # Save next to the normalized raster, same basename + _radcal.png
        plot_path = os.path.join(
            os.path.dirname(os.path.abspath(outfn)),
            os.path.splitext(os.path.basename(outfn))[0] + '_radcal.png'
        )
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'radcal plot saved to: {plot_path}')
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
    options, args = getopt.getopt(sys.argv[1:], 'hnp:d:', ['ncp-threshold='])
    pos = None
    dims = None
    ncp_threshold = 0.95
    fsfn = None
    graphics = True
    for option, value in options:
        if option == '-h':
            print(usage)
            sys.exit()
        elif option == '-n':
            graphics = False
        elif option == '-p':
            pos = ast.literal_eval(value)
        elif option == '-d':
            dims = ast.literal_eval(value)
        elif option == '--ncp-threshold':
            ncp_threshold = float(value)
    if (len(args) != 1) and (len(args) != 2):
        print('Incorrect number of arguments')
        print(usage)
        sys.exit(1)
    imadfn = args[0]
    if len(args) == 2:
        fsfn = args[1]

    main(imadfn, ncp_threshold, pos, dims, fsfn, graphics=graphics)
