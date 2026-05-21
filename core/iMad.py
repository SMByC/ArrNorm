#!/usr/bin/env python3
# ******************************************************************************
#  Name:     iMad.py
#  Purpose:  Iteratively Reweighted Multivariate Alteration Detection
#            (IR-MAD) for bi-temporal multispectral imagery.
#
#  Original algorithm: M. J. Canty (2014). Refactored for numerical
#  stability and performance: see comments inline.
#
#  License: GPLv2+
# ******************************************************************************

import os
import sys
import time
from operator import itemgetter

import numpy as np
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly, GDT_Float32
from scipy import stats

import core.auxil.auxil as auxil

# Block height (in rows) used when streaming both images band-by-band into
# the running covariance accumulator. Reading 256 rows at a time amortizes
# the per-call GDAL overhead by ~256x vs the original row-by-row loop
# while keeping peak memory bounded (256 * cols * 2*bands * 8 bytes).
DEFAULT_BLOCK_ROWS = 256


def _iter_row_blocks(rows, block_rows):
    """Yield (y_offset, n_rows) chunks covering [0, rows)."""
    for y in range(0, rows, block_rows):
        yield y, min(block_rows, rows - y)


def _read_block(raster_bands, x0, y0, cols, n_rows):
    """Read a (n_rows, cols, bands) float64 tile for the given band list."""
    bands = len(raster_bands)
    tile = np.empty((n_rows * cols, bands), dtype=np.float64)
    for k, rb in enumerate(raster_bands):
        arr = rb.ReadAsArray(x0, y0, cols, n_rows)
        tile[:, k] = np.nan_to_num(arr, copy=False).ravel()
    return tile


def main(img_ref, img_target, max_iters=30, conv_threshold=0.99, band_pos=None, dims=None,
         graphics=False, ref_text='', block_rows=DEFAULT_BLOCK_ROWS):
    gdal.AllRegister()
    start = time.time()  # was previously undefined at print-elapsed time (bug)

    path = os.path.dirname(os.path.abspath(img_ref))
    basename1 = os.path.basename(img_ref)
    root1, ext1 = os.path.splitext(basename1)
    basename2 = os.path.basename(img_target)
    root2, _ext2 = os.path.splitext(basename2)
    outfn = os.path.join(path, f'MAD({root1}&{basename2}){ext1}')

    inDataset1 = gdal.Open(img_ref, GA_ReadOnly)
    inDataset2 = gdal.Open(img_target, GA_ReadOnly)
    if inDataset1 is None or inDataset2 is None:
        sys.stderr.write("Error: input image(s) could not be opened.\n")
        sys.exit(1)

    cols = inDataset1.RasterXSize
    rows = inDataset1.RasterYSize
    bands = inDataset1.RasterCount
    cols2 = inDataset2.RasterXSize
    rows2 = inDataset2.RasterYSize
    bands2 = inDataset2.RasterCount

    if bands != bands2:
        sys.stderr.write(
            f"Band count mismatch between reference ({bands}) "
            f"and target ({bands2}).\n")
        sys.exit(1)

    if band_pos is None:
        band_pos = list(range(1, bands + 1))
    else:
        bands = len(band_pos)

    if dims is None:
        x0 = y0 = 0
    else:
        x0, y0, cols, rows = dims

    # If the target was warped during registration we assume it is already
    # aligned to the reference origin; otherwise use the same window.
    if root2.find('_warp') != -1:
        x2 = y2 = 0
    else:
        x2, y2 = x0, y0

    # Dimension guard: after clipper() in bin/arrnorm, both images MUST share
    # the same pixel grid. Pixel-for-pixel alignment is required for IR-MAD.
    if cols != cols2 or rows != rows2:
        raise Exception(
            f"\n ERROR: Reference clip ({cols}x{rows}) and target "
            f"({cols2}x{rows2}) have different pixel dimensions.\n"
            f" Pixel-for-pixel alignment is required for IR-MAD. Ensure both "
            f"images share the same CRS, pixel size, and spatial extent "
            f"before running the normalization.\n")

    print('------------IRMAD -------------')
    rasterBands1 = [inDataset1.GetRasterBand(b) for b in band_pos]
    rasterBands2 = [inDataset2.GetRasterBand(b) for b in band_pos]

    # Sanity-check: any band that is entirely zero would make the algorithm
    # degenerate (singular covariance). Bail out early with a clear message.
    for k, rb in enumerate(rasterBands1):
        if not rb.ReadAsArray().any():
            print(f"\nERROR: band {band_pos[k]} of '{basename1}' has only "
                  f"zeros — please check it.\n")
            sys.exit(1)
    for k, rb in enumerate(rasterBands2):
        if not rb.ReadAsArray().any():
            print(f"\nERROR: band {band_pos[k]} of '{basename2}' has only "
                  f"zeros — please check it.\n")
            sys.exit(1)

    cpm = auxil.Cpm(2 * bands)
    oldrho = np.zeros(bands)
    rhos = np.zeros((max_iters, bands))
    results = []
    sigMADs = means1 = means2 = A = B = None

    delta_thres = 1.0 - conv_threshold
    print(f'\nStop condition: max iterations ({max_iters}) or delta < {round(delta_thres, 5)}\n'
          f'with auto selection of the best delta for the final result:')
    print(f' {ref_text + " ->"} iteration: 0, delta: 1.0 ({time.asctime()})')

    current_iter = 0
    while current_iter < max_iters:
        try:
            # ---- pass 1: accumulate weighted covariance over the full image
            for ry, nr in _iter_row_blocks(rows, block_rows):
                tile_ref = _read_block(rasterBands1, x0, y0 + ry, cols, nr)
                tile_tgt = _read_block(rasterBands2, x2, y2 + ry, cols, nr)
                tile = np.concatenate((tile_ref, tile_tgt), axis=1)

                # Exclude rows where any image has a fully-zero pixel
                # (treated as no-data) — preserves the original behaviour.
                nz_ref = tile_ref.any(axis=1)
                nz_tgt = tile_tgt.any(axis=1)
                keep = nz_ref & nz_tgt

                if current_iter > 0:
                    # MAD variates and chi-square statistic for weighting
                    mads = ((tile[:, 0:bands] - means1[0]) @ A
                            - (tile[:, bands:] - means2[0]) @ B)
                    chisqr = np.sum((mads / sigMADs[0]) ** 2, axis=1)
                    # chi2.sf == 1 - chi2.cdf, but stable in the upper tail
                    wts = stats.chi2.sf(chisqr, bands)
                    cpm.update(tile[keep], wts[keep])
                else:
                    cpm.update(tile[keep])

            # ---- canonical-correlation step
            S = cpm.covariance()
            means = cpm.means()
            cpm.reset()

            s11 = S[0:bands, 0:bands]
            s22 = S[bands:, bands:]
            s12 = S[0:bands, bands:]
            s21 = s12.T  # S is symmetric

            # Solve the two coupled generalized eigenproblems
            #   s12 s22^-1 s21  a = mu^2  s11  a
            #   s21 s11^-1 s12  b = mu^2  s22  b
            if bands > 1:
                # scipy.linalg.solve is more stable than forming inv() explicitly
                from scipy.linalg import solve
                c1 = s12 @ solve(s22, s21, assume_a='pos')
                c2 = s21 @ solve(s11, s12, assume_a='pos')
                mu2a, A = auxil.geneiv(c1, s11)
                mu2b, B = auxil.geneiv(c2, s22)
                idx_a = np.argsort(mu2a)
                idx_b = np.argsort(mu2b)
                A = A[:, idx_a]
                B = B[:, idx_b]
                mu2 = mu2b[idx_b]
            else:
                mu2 = (s12 * s21 / s22) / s11
                A = np.array([[1.0 / np.sqrt(s11[0, 0])]])
                B = np.array([[1.0 / np.sqrt(s22[0, 0])]])

            # Clamp to [0, 1] before sqrt — round-off can push mu^2 slightly
            # negative or slightly above 1, which would yield NaN.
            mu2 = np.clip(mu2, 0.0, 1.0)
            rho = np.sqrt(mu2)
            sigma = np.sqrt(2.0 * (1.0 - rho))  # std of each MAD variate
            delta = float(np.max(np.abs(rho - oldrho)))

            rhos[current_iter, :] = rho
            oldrho = rho

            # Tile sigma and means to (1, ...) — broadcast over (n_pixels, bands)
            sigMADs = sigma[None, :]
            means1 = means[None, 0:bands]
            means2 = means[None, bands:]

            # Sign-fix: ensure each canonical variate has a positive sum of
            # correlations with the X channels (otherwise eigenvectors can
            # flip sign between iterations, breaking the stopping criterion).
            D = 1.0 / np.sqrt(np.diag(s11))            # vector form of diag(D)
            sgn_a = np.sign(np.sum(D[:, None] * s11 @ A, axis=0))
            sgn_a[sgn_a == 0] = 1.0
            A = A * sgn_a
            sgn_cov = np.sign(np.diag(A.T @ s12 @ B))
            sgn_cov[sgn_cov == 0] = 1.0
            B = B * sgn_cov

            current_iter += 1
            print(f' {ref_text + " ->"} iteration: {current_iter}, '
                  f'delta: {round(delta, 5)} ({time.asctime()})')
            results.append((delta, {"iter": current_iter, "A": A, "B": B,
                                    "means1": means1, "means2": means2,
                                    "sigMADs": sigMADs, "rho": rho}))

            # Convergence check: stop when the maximum change in canonical
            # correlations falls below the threshold derived from conv_threshold.
            # Skip on the first iteration because oldrho starts at zero,
            # making delta a magnitude estimate rather than a convergence measure.
            if current_iter > 1 and delta < delta_thres:
                print(f' Convergence reached at iteration {current_iter} '
                      f'(delta={round(delta, 5)} < {round(delta_thres, 5)})')
                break

        except Exception as err:
            print(f"\n WARNING: exception at iteration {current_iter}: {err}\n"
                  f" Falling back to best-delta result computed so far. "
                  f"Verify the input bands.\n")
            current_iter = max_iters  # exit the while-loop

        if current_iter == max_iters:
            # Pick the iteration with the smallest delta — the run with the
            # most-converged canonical correlations.
            best = sorted(results, key=itemgetter(0))[0]
            print(f"\n Best delta over all iterations: {round(best[0], 5)} "
                  f"(iteration {best[1]['iter']}). "
                  f"Final result computed with those parameters.")
            delta = best[0]
            A = best[1]["A"]
            B = best[1]["B"]
            means1 = best[1]["means1"]
            means2 = best[1]["means2"]
            sigMADs = best[1]["sigMADs"]
            rho = best[1]["rho"]
            del results

    print(f'\nRHO: {rho}')

    # ---- write MAD variates + chi-square band to disk
    driver = inDataset1.GetDriver()
    outDataset = driver.Create(outfn, cols, rows, bands + 1, GDT_Float32)
    projection = inDataset1.GetProjection()
    geotransform = inDataset1.GetGeoTransform()
    if geotransform is not None:
        gt = list(geotransform)
        gt[0] = gt[0] + x0 * gt[1]
        gt[3] = gt[3] + y0 * gt[5]
        outDataset.SetGeoTransform(tuple(gt))
    if projection is not None:
        outDataset.SetProjection(projection)
    outBands = [outDataset.GetRasterBand(k + 1) for k in range(bands + 1)]

    for ry, nr in _iter_row_blocks(rows, block_rows):
        tile_ref = _read_block(rasterBands1, x0, y0 + ry, cols, nr)
        tile_tgt = _read_block(rasterBands2, x2, y2 + ry, cols, nr)
        mads = (tile_ref - means1[0]) @ A - (tile_tgt - means2[0]) @ B
        chisqr = np.sum((mads / sigMADs[0]) ** 2, axis=1)
        for k in range(bands):
            outBands[k].WriteArray(mads[:, k].reshape(nr, cols), 0, ry)
        outBands[bands].WriteArray(chisqr.reshape(nr, cols), 0, ry)
    for outBand in outBands:
        outBand.FlushCache()
    outDataset = None
    inDataset1 = None
    inDataset2 = None

    print('result written to: ' + outfn)
    print(f'elapsed time: {time.time() - start:.2f}s')

    if graphics:
        try:
            import matplotlib.pyplot as plt
            x = np.arange(current_iter)
            plt.plot(x, rhos[:current_iter, :])
            plt.title('Canonical correlations')
            plt.show()
        except ImportError:
            pass  # matplotlib not available; skip graphics

    return outfn
