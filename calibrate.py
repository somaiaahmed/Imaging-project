"""
calibrate.py
============
Stage 1 Beam-Hardening Polynomial Calibration.

Theory
------
In a polychromatic CT system, the measured line integral p_BH is a
nonlinear (concave) function of the true monochromatic line integral p:

    p_BH ≈ a1·p + a2·p² + a3·p³ + …

The correction inverts this: given p_BH we want to recover p.

Calibration strategy
--------------------
We have access to:
  • sino_ideal.pj  – the "ground truth" monochromatic sinogram
  • sino_BH.pj     – the beam-hardened (corrupted) sinogram

Both share the same geometry, so pixel (i, j) in one corresponds
exactly to pixel (i, j) in the other.  We fit a polynomial

    p_ideal ≈ f(p_BH) = c0 + c1·p_BH + c2·p_BH² + … + cn·p_BH^n

by ordinary least-squares regression over all pixels.

In a real scanner without a reference scan, the same curve would be
estimated from water-phantom measurements.  The algebra is identical.

Outputs
-------
  calibration.npz   – polynomial coefficients + fit diagnostics
"""

import numpy as np
from pj_io import read_pj

# ── Config ───────────────────────────────────────────────────────────────────
NVIEW          = 360
NDET           = 512
POLY_DEGREE    = 3          # degree of the correction polynomial
IDEAL_FILE     = "sino_ideal.pj"
BH_FILE        = "sino_BH.pj"
OUT_FILE       = "calibration.npz"

# Subsample for speed (every N-th pixel); set to 1 to use all pixels
SUBSAMPLE      = 4


def build_calibration(ideal_file=IDEAL_FILE,
                      bh_file=BH_FILE,
                      degree=POLY_DEGREE,
                      subsample=SUBSAMPLE,
                      out_file=OUT_FILE):
    """
    Fit a polynomial p_ideal = f(p_BH) and save the coefficients.

    Returns
    -------
    coeffs : np.ndarray, shape (degree+1,)
        Polynomial coefficients in *increasing* power order:
        [c0, c1, c2, …, c_degree]  (np.polyfit returns *decreasing* order;
        we flip so index == power for clarity).
    diagnostics : dict
    """
    print("=== Stage 1 BH Calibration ===\n")

    # ── Load sinograms ───────────────────────────────────────────────────────
    print(f"Loading ideal sinogram  : {ideal_file}")
    p_ideal, _ = read_pj(ideal_file, nview=NVIEW, ndet=NDET)

    print(f"\nLoading BH sinogram     : {bh_file}")
    p_bh,   _  = read_pj(bh_file,   nview=NVIEW, ndet=NDET)

    # ── Flatten + subsample ──────────────────────────────────────────────────
    p_ideal_flat = p_ideal.ravel()[::subsample]
    p_bh_flat    = p_bh.ravel()[::subsample]
    n_samples    = p_ideal_flat.size
    print(f"\nCalibration samples     : {n_samples:,}  (subsample={subsample})")

    # Only use pixels where both sinograms have meaningful signal
    # (skip near-zero air paths where the ratio is degenerate)
    mask = (p_bh_flat > 0.01) & (p_ideal_flat > 0.01)
    x = p_bh_flat[mask]
    y = p_ideal_flat[mask]
    print(f"After threshold mask    : {x.size:,} samples")

    # ── Polynomial fit ───────────────────────────────────────────────────────
    # np.polyfit returns coefficients in *decreasing* power order
    poly_dec = np.polyfit(x, y, deg=degree)
    poly_fn  = np.poly1d(poly_dec)

    # Flip to increasing-power order for clarity / storage
    coeffs = poly_dec[::-1]

    # ── Diagnostics ──────────────────────────────────────────────────────────
    y_pred = poly_fn(x)
    residuals = y - y_pred
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2     = 1.0 - ss_res / ss_tot
    rmse   = float(np.sqrt(np.mean(residuals ** 2)))
    max_err = float(np.max(np.abs(residuals)))

    print(f"\n── Fit quality (degree={degree}) ──")
    print(f"  R²       : {r2:.6f}  (1.0 = perfect)")
    print(f"  RMSE     : {rmse:.6f}")
    print(f"  Max |err|: {max_err:.6f}")
    print(f"\n── Coefficients (c0 + c1·x + c2·x² + …) ──")
    for i, c in enumerate(coeffs):
        print(f"  c{i} = {c:+.8f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    diagnostics = {"r2": r2, "rmse": rmse, "max_err": max_err,
                   "degree": degree, "n_samples": x.size}
    np.savez(out_file,
             coeffs=coeffs,
             degree=np.array(degree),
             r2=np.array(r2),
             rmse=np.array(rmse))
    print(f"\nCalibration saved → {out_file}")

    return coeffs, diagnostics


if __name__ == "__main__":
    build_calibration()
