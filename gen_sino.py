"""
gen_sino.py
===========
Generate a Shepp-Logan phantom, simulate parallel-beam projections at
360 angles, apply a beam-hardening (BH) polynomial correction, and save
both the ideal and BH-corrupted sinograms as .pj files.

Usage
-----
    python3 gen_sino.py

Outputs
-------
    sino_ideal.pj   – clean (Beer-Lambert) sinogram
    sino_BH.pj      – sinogram with beam-hardening distortion
"""

import numpy as np
from pj_io import write_pj

# ── Optional: use scikit-image for the Radon transform if available ─────────
try:
    from skimage.transform import radon, iradon
    _HAVE_SKIMAGE = True
except ImportError:
    _HAVE_SKIMAGE = False
    print("scikit-image not found – using a simple manual Radon transform.")


# ════════════════════════════════════════════════════════════════════════════
# 1.  Shepp-Logan phantom
# ════════════════════════════════════════════════════════════════════════════

def shepp_logan(n=512):
    """
    Return an n×n Shepp-Logan phantom (float32, values in [0, 1]).
    Defined by the 10 standard ellipses from Herman & Lakshminarayanan 1980.
    """
    # Each row: [amplitude, a, b, x0, y0, phi_degrees]
    ellipses = [
        [ 1.0,   .6900,  .9200,  .00,   .00,   0],
        [-.8,    .6624,  .8740,  .00,  -.0184,  0],
        [-.2,    .1100,  .3100,  .22,   .00,  -18],
        [-.2,    .1600,  .4100, -.22,   .00,   18],
        [ .1,    .2100,  .2500,  .00,   .35,   0],
        [ .1,    .0460,  .0460,  .00,   .10,   0],
        [ .1,    .0460,  .0460,  .00,  -.10,   0],
        [ .1,    .0460,  .0230, -.08,  -.605,  0],
        [ .1,    .0230,  .0230,  .00,  -.606,  0],
        [ .1,    .0230,  .0460,  .06,  -.605,  0],
    ]

    phantom = np.zeros((n, n), dtype=np.float32)
    # Pixel coordinates in [−1, 1]
    coords = np.linspace(-1, 1, n)
    x, y = np.meshgrid(coords, coords)

    for amp, a, b, x0, y0, phi in ellipses:
        phi_r = np.deg2rad(phi)
        cp, sp = np.cos(phi_r), np.sin(phi_r)
        xr = (x - x0) * cp + (y - y0) * sp
        yr = -(x - x0) * sp + (y - y0) * cp
        mask = (xr / a) ** 2 + (yr / b) ** 2 <= 1.0
        phantom[mask] += amp

    phantom = np.clip(phantom, 0, None)
    phantom /= phantom.max()
    return phantom


# ════════════════════════════════════════════════════════════════════════════
# 2.  Radon transform (parallel-beam forward projection)
# ════════════════════════════════════════════════════════════════════════════

def forward_project(phantom, angles_deg):
    """
    Return sinogram (n_angles × n_det) via Radon transform.
    Uses scikit-image when available, otherwise a simple line-integral loop.
    """
    if _HAVE_SKIMAGE:
        # radon returns (ndet, nangles) – transpose to (nangles, ndet)
        sino = radon(phantom, theta=angles_deg, circle=True).T
        # Normalize to [0, 1]
        sino = sino.astype(np.float32)
        sino -= sino.min()
        if sino.max() > 0:
            sino /= sino.max()
        return sino
    else:
        return _manual_radon(phantom, angles_deg)


def _manual_radon(phantom, angles_deg):
    n = phantom.shape[0]
    ndet = n
    sino = np.zeros((len(angles_deg), ndet), dtype=np.float32)
    coords = np.arange(ndet) - ndet // 2          # detector coords

    for i, ang in enumerate(angles_deg):
        theta = np.deg2rad(ang)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        # Project along the perpendicular direction
        for d_idx, d in enumerate(coords):
            # Line: all pixels where x*cos+y*sin ≈ d (nearest-neighbour)
            col = int(round(d * cos_t + n // 2))
            row = int(round(-d * sin_t + n // 2))
            # Sum along the ray (simple column/row slice)
            if 0 <= col < n:
                sino[i, d_idx] = phantom[:, col].sum() / n
    return sino


# ════════════════════════════════════════════════════════════════════════════
# 3.  Beam-hardening simulation
# ════════════════════════════════════════════════════════════════════════════

def apply_beam_hardening(sino, order=3, coeffs=None):
    """
    Simulate polychromatic beam hardening by applying a polynomial
    distortion to the (already log-transformed) line integrals.

    A typical BH model:
        sino_BH[i] = a1*sino[i] + a2*sino[i]^2 + a3*sino[i]^3

    The polynomial makes thin paths look almost the same as the ideal,
    but causes cupping / streaks for thick paths.

    Parameters
    ----------
    sino : ndarray, float32, values in [0, 1]
    order : int
        Polynomial degree (default 3).
    coeffs : list of float or None
        Polynomial coefficients [a1, a2, …, a_order].
        Default: [0.95, 0.08, -0.03] (typical soft-tissue BH curve).

    Returns
    -------
    sino_BH : ndarray, same shape and dtype as sino
    """
    if coeffs is None:
        # Slightly sub-linear → cupping effect in reconstructed image
        coeffs = [0.95, 0.08, -0.03][:order]

    sino_BH = np.zeros_like(sino)
    for power, coeff in enumerate(coeffs, start=1):
        sino_BH += coeff * (sino ** power)

    # Re-normalise to [0, 1] so the writer doesn't clip anything
    sino_BH = np.clip(sino_BH, 0, None)
    if sino_BH.max() > 0:
        sino_BH /= sino_BH.max()

    return sino_BH.astype(np.float32)


# ════════════════════════════════════════════════════════════════════════════
# 4.  Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    NVIEW = 360
    NDET  = 512
    N_PHM = 512        # phantom pixel size (square)

    print("=== Shepp-Logan phantom generation ===")
    phantom = shepp_logan(N_PHM)
    print(f"Phantom shape : {phantom.shape}, min={phantom.min():.3f}, max={phantom.max():.3f}")

    angles = np.linspace(0, 180, NVIEW, endpoint=False)   # 0 … 179.5°

    print("\n=== Forward projection (Radon transform) ===")
    sino_ideal = forward_project(phantom, angles)

    # Resize detector dimension to NDET if needed
    if sino_ideal.shape[1] != NDET:
        from scipy.ndimage import zoom
        sino_ideal = zoom(sino_ideal, (1, NDET / sino_ideal.shape[1]), order=1)
        sino_ideal = sino_ideal.astype(np.float32)

    print(f"Sinogram shape : {sino_ideal.shape}")
    print(f"Min / Max      : {sino_ideal.min():.6f} / {sino_ideal.max():.6f}")

    print("\n=== Applying beam hardening ===")
    sino_BH = apply_beam_hardening(sino_ideal, order=3)
    print(f"BH sinogram: Min / Max : {sino_BH.min():.6f} / {sino_BH.max():.6f}")

    print("\n=== Writing .pj files ===")
    write_pj("sino_ideal.pj", sino_ideal)
    write_pj("sino_BH.pj",    sino_BH)

    print("\nDone.  Files written:")
    print("  sino_ideal.pj  – clean sinogram")
    print("  sino_BH.pj     – beam-hardened sinogram")


if __name__ == "__main__":
    main()
