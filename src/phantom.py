"""
phantom.py
==========
Shepp-Logan phantom generation, parallel-beam forward projection,
and polychromatic beam-hardening simulation.

Classes
-------
SheppLoganPhantom       Generates the standard 10-ellipse phantom.
ForwardProjector        Radon-transform wrapper (scikit-image or fallback).
BeamHardeningSimulator  Applies a polynomial BH distortion to a sinogram.
"""

from __future__ import annotations

import numpy as np

try:
    from skimage.transform import radon
    _HAVE_SKIMAGE = True
except ImportError:
    _HAVE_SKIMAGE = False
    print("scikit-image not found – using built-in Radon fallback.")


# ── Shepp-Logan Phantom ────────────────────────────────────────────────────────

class SheppLoganPhantom:
    """
    Standard 10-ellipse Shepp-Logan phantom.

    Parameters
    ----------
    size : int
        Side length of the square output array (default 512).
    """

    # [amplitude, a, b, x0, y0, phi_degrees]
    _ELLIPSES = [
        [ 1.0,  .6900, .9200,  .00,   .00,   0],
        [-.80,  .6624, .8740,  .00,  -.0184, 0],
        [-.20,  .1100, .3100,  .22,   .00,  -18],
        [-.20,  .1600, .4100, -.22,   .00,   18],
        [ .10,  .2100, .2500,  .00,   .35,   0],
        [ .10,  .0460, .0460,  .00,   .10,   0],
        [ .10,  .0460, .0460,  .00,  -.10,   0],
        [ .10,  .0460, .0230, -.08,  -.605,  0],
        [ .10,  .0230, .0230,  .00,  -.606,  0],
        [ .10,  .0230, .0460,  .06,  -.605,  0],
    ]

    def __init__(self, size: int = 512):
        self.size = size
        self.image: np.ndarray = self._generate()

    def _generate(self) -> np.ndarray:
        n = self.size
        phantom = np.zeros((n, n), dtype=np.float32)
        coords = np.linspace(-1, 1, n)
        x, y = np.meshgrid(coords, coords)

        for amp, a, b, x0, y0, phi in self._ELLIPSES:
            phi_r = np.deg2rad(phi)
            cp, sp = np.cos(phi_r), np.sin(phi_r)
            xr = (x - x0) * cp + (y - y0) * sp
            yr = -(x - x0) * sp + (y - y0) * cp
            mask = (xr / a) ** 2 + (yr / b) ** 2 <= 1.0
            phantom[mask] += amp

        phantom = np.clip(phantom, 0, None)
        phantom /= phantom.max()
        return phantom

    def __repr__(self) -> str:
        return f"SheppLoganPhantom(size={self.size})"


# ── Forward Projector ──────────────────────────────────────────────────────────

class ForwardProjector:
    """
    Parallel-beam forward projector (Radon transform).

    Uses scikit-image when available; falls back to a simple
    nearest-neighbour column-sum projection otherwise.

    Parameters
    ----------
    n_views : int
        Number of projection angles (default 360).
    n_det : int
        Number of detector bins (default 512).
    """

    def __init__(self, n_views: int = 360, n_det: int = 512):
        self.n_views = n_views
        self.n_det   = n_det
        self.angles  = np.linspace(0, 180, n_views, endpoint=False)

    def project(self, phantom: np.ndarray) -> np.ndarray:
        """
        Forward project a 2-D phantom into a sinogram.

        Parameters
        ----------
        phantom : np.ndarray, shape (N, N)

        Returns
        -------
        sino : np.ndarray, shape (n_views, n_det), float32, range [0, 1]
        """
        if _HAVE_SKIMAGE:
            sino = radon(phantom, theta=self.angles, circle=True).T
        else:
            sino = self._manual_radon(phantom)

        sino = sino.astype(np.float32)
        sino -= sino.min()
        if sino.max() > 0:
            sino /= sino.max()

        # Resize detector axis if needed
        if sino.shape[1] != self.n_det:
            from scipy.ndimage import zoom
            sino = zoom(sino, (1, self.n_det / sino.shape[1]), order=1).astype(np.float32)

        return sino

    def _manual_radon(self, phantom: np.ndarray) -> np.ndarray:
        n    = phantom.shape[0]
        sino = np.zeros((self.n_views, n), dtype=np.float32)
        coords = np.arange(n) - n // 2

        for i, ang in enumerate(self.angles):
            theta = np.deg2rad(ang)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            for d_idx, d in enumerate(coords):
                col = int(round(d * cos_t + n // 2))
                if 0 <= col < n:
                    sino[i, d_idx] = phantom[:, col].sum() / n
        return sino


# ── Beam Hardening Simulator ───────────────────────────────────────────────────

class BeamHardeningSimulator:
    """
    Simulates polychromatic beam hardening via a polynomial distortion.

    The model:
        sino_BH = a1·sino + a2·sino² + a3·sino³ + …

    Parameters
    ----------
    order : int
        Polynomial degree (default 3).
    coeffs : list[float] or None
        Per-power coefficients [a1, a2, …].
        Defaults to [0.95, 0.08, -0.03] (mild soft-tissue BH).
    """

    _DEFAULT_COEFFS = [0.95, 0.08, -0.03]

    def __init__(self, order: int = 3, coeffs: list[float] | None = None):
        self.order  = order
        self.coeffs = (coeffs or self._DEFAULT_COEFFS)[:order]

    def apply(self, sino: np.ndarray) -> np.ndarray:
        """
        Apply beam hardening to a normalised sinogram.

        Parameters
        ----------
        sino : np.ndarray, float32, values in [0, 1]

        Returns
        -------
        sino_BH : np.ndarray, same shape, float32, range [0, 1]
        """
        sino_bh = np.zeros_like(sino)
        for power, coeff in enumerate(self.coeffs, start=1):
            sino_bh += coeff * (sino ** power)

        sino_bh = np.clip(sino_bh, 0, None)
        if sino_bh.max() > 0:
            sino_bh /= sino_bh.max()
        return sino_bh.astype(np.float32)