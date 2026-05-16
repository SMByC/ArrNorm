#!/usr/bin/env python3
# ******************************************************************************
#  Name:     auxil.py
#  Purpose:  Math primitives used by the IR-MAD / RadCal / Register pipeline.
#
#  Only four symbols are exported:
#     Cpm           -- weighted streaming mean / covariance accumulator
#     geneiv        -- symmetric generalized eigenproblem  A x = lambda B x
#     orthoregress  -- orthogonal (total-least-squares) regression
#     similarity    -- log-polar Fourier image-image similarity transform
#
#  Original auxiliaries: M. Canty 2012 (DWT, ATWT, PCA, MNF, kernels, PNG
#  output, ENVI header parsing, supervised classifiers, congrid, ...) were
#  not used by the IR-MAD pipeline and have been removed.
#
#  License: GPLv2+
# ******************************************************************************

import math

import numpy as np
import scipy.linalg
import scipy.ndimage as ndii
from numpy.fft import fft2, fftshift, ifft2


# -----------------
# provisional means
# -----------------

class Cpm(object):
    """Weighted running mean / cross-product accumulator.

    Pure-numpy replacement for the original ctypes 'provmeans' shared
    library. Tracks the weighted mean and the weighted sum of squared/
    cross-product deviations (SSCP) for an N-variate stream of
    observations, using West's algorithm (numerically stable single
    pass).

    With incoming batch x_i and weights w_i:
        SW_new   = SW + sum(w_i)
        delta_i  = x_i - mean_old
        mean_new = mean_old + sum(w_i * delta_i) / SW_new
        cov_new  = cov_old + sum(w_i * delta_i * (x_i - mean_new).T)
    covariance() returns cov_new / (SW - 1).
    """

    def __init__(self, N):
        self.N = N
        self.reset()

    def reset(self):
        self.mn = np.zeros(self.N)
        self.cov = np.zeros((self.N, self.N))
        self.sw = 1e-7  # tiny seed avoids divide-by-zero on covariance()

    def update(self, Xs, Ws=None):
        Xs = np.asarray(Xs, dtype=np.float64)
        if Xs.ndim == 1:
            Xs = Xs.reshape(1, -1)
        n = Xs.shape[0]
        if n == 0:
            return
        if Ws is None:
            Ws = np.ones(n, dtype=np.float64)
        else:
            Ws = np.asarray(Ws, dtype=np.float64)

        sw_new = self.sw + Ws.sum()
        delta = Xs - self.mn                              # (n, N)
        weighted_delta = Ws[:, None] * delta              # (n, N)
        self.mn = self.mn + weighted_delta.sum(axis=0) / sw_new
        delta2 = Xs - self.mn                             # (n, N) with NEW mean
        self.cov = self.cov + weighted_delta.T @ delta2   # (N, N)
        self.sw = sw_new

    def covariance(self):
        c = self.cov / (self.sw - 1.0)
        return 0.5 * (c + c.T)  # symmetrize tiny asymmetric drift

    def means(self):
        return self.mn


# ---------------------------------
# symmetric generalized eigenproblem
# ---------------------------------

def geneiv(A, B):
    """Solve the symmetric generalized eigenproblem  A x = lambda B x.

    Returns (eigenvalues, eigenvectors-in-columns), eigenvalues sorted
    ascending. Uses scipy.linalg.eigh which dispatches to LAPACK's
    DSYGVD: it handles the Cholesky reduction internally with a single
    call and is more numerically stable than building inv(chol(B))
    explicitly.
    """
    A_s = 0.5 * (A + A.T)
    B_s = 0.5 * (B + B.T)
    return scipy.linalg.eigh(A_s, B_s)


# ---------------------
# orthogonal regression
# ---------------------

def orthoregress(x, y):
    """Total-least-squares (orthogonal) regression of y on x.

    Returns (slope, intercept, Pearson R). Uses the closed-form
    solution for the 2-variable case (major axis of the (x, y)
    covariance ellipse):
        b = (Syy - Sxx + sqrt((Syy - Sxx)^2 + 4*Sxy^2)) / (2*Sxy)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xm = x.mean()
    ym = y.mean()
    dx = x - xm
    dy = y - ym
    n = x.size - 1
    sxx = np.dot(dx, dx) / n
    syy = np.dot(dy, dy) / n
    sxy = np.dot(dx, dy) / n
    denom = math.sqrt(sxx * syy)
    R = sxy / denom if denom > 0.0 else 0.0
    if sxy == 0.0:
        return [0.0, ym, R]
    b = (syy - sxx + math.sqrt((syy - sxx) ** 2 + 4.0 * sxy * sxy)) / (2.0 * sxy)
    return [b, ym - b * xm, R]


# -----------------------------
# image-image similarity (Fourier log-polar)
# -----------------------------

def similarity(bn0, bn1):
    """Estimate (scale, angle, [t_row, t_col]) registering bn1 -> bn0.

    Uses the Reddy-Chatterji log-polar / Fourier cross-correlation
    method: scale and rotation are recovered from the magnitude
    spectrum in log-polar coordinates, then translation is recovered
    from phase correlation of the rectified images.

    Adapted from M. Canty 2012 / Christoph Gohlke's Imreg.py.
    """

    def highpass(shape):
        """High-pass cosine filter to suppress DC before log-polar mapping."""
        x = np.outer(
            np.cos(np.linspace(-math.pi / 2., math.pi / 2., shape[0])),
            np.cos(np.linspace(-math.pi / 2., math.pi / 2., shape[1])))
        return (1.0 - x) * (2.0 - x)

    def logpolar(image, angles=None, radii=None):
        """Map `image` into log-polar coordinates, return (image, log_base)."""
        shape = image.shape
        center = shape[0] / 2, shape[1] / 2
        if angles is None:
            angles = shape[0]
            if radii is None:
                radii = shape[1]
        theta = np.empty((angles, radii), dtype=np.float64)
        theta.T[:] = -np.linspace(0, np.pi, angles, endpoint=False)
        d = np.hypot(shape[0] - center[0], shape[1] - center[1])
        log_base = 10.0 ** (math.log10(d) / radii)
        radius = np.empty_like(theta)
        radius[:] = np.power(log_base, np.arange(radii, dtype=np.float64)) - 1.0
        x = radius * np.sin(theta) + center[0]
        y = radius * np.cos(theta) + center[1]
        output = np.empty_like(x)
        ndii.map_coordinates(image, [x, y], output=output)
        return output, log_base

    lines0, samples0 = bn0.shape
    bn1 = bn1[0:lines0, 0:samples0]  # crop to reference shape

    # ---- scale + angle from log-polar of magnitude spectra
    f0 = fftshift(abs(fft2(bn0)))
    f1 = fftshift(abs(fft2(bn1)))
    h = highpass(f0.shape)
    f0 *= h
    f1 *= h
    del h
    f0, log_base = logpolar(f0)
    f1, log_base = logpolar(f1)
    f0 = fft2(f0)
    f1 = fft2(f1)
    r0 = abs(f0) * abs(f1)
    ir = abs(ifft2((f0 * f1.conjugate()) / r0))
    i0, i1 = np.unravel_index(np.argmax(ir), ir.shape)
    angle = 180.0 * i0 / ir.shape[0]
    scale = log_base ** i1
    if scale > 1.8:
        # try the inverse direction
        ir = abs(ifft2((f1 * f0.conjugate()) / r0))
        i0, i1 = np.unravel_index(np.argmax(ir), ir.shape)
        angle = -180.0 * i0 / ir.shape[0]
        scale = 1.0 / (log_base ** i1)
        if scale > 1.8:
            raise ValueError("Images are not compatible. Scale change > 1.8")
    if angle < -90.0:
        angle += 180.0
    elif angle > 90.0:
        angle -= 180.0

    # ---- translation from phase-correlation of the rectified image
    bn2 = ndii.zoom(bn1, 1.0 / scale)
    bn2 = ndii.rotate(bn2, angle)
    if bn2.shape < bn0.shape:
        t = np.zeros_like(bn0)
        t[:bn2.shape[0], :bn2.shape[1]] = bn2
        bn2 = t
    elif bn2.shape > bn0.shape:
        bn2 = bn2[:bn0.shape[0], :bn0.shape[1]]
    f0 = fft2(bn0)
    f1 = fft2(bn2)
    ir = abs(ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1))))
    t0, t1 = np.unravel_index(np.argmax(ir), ir.shape)
    if t0 > f0.shape[0] // 2:
        t0 -= f0.shape[0]
    if t1 > f0.shape[1] // 2:
        t1 -= f0.shape[1]
    return (scale, angle, [t0, t1])
