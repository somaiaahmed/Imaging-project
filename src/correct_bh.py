"""
correct_bh.py
=============
Stage 1 Beam-Hardening Polynomial Correction.

Loads the calibration polynomial from calibration.npz and applies it
pixel-wise to sino_BH.pj, producing sino_corrected.pj.

Usage
-----
    # Step 1 – build calibration curve (only once)
    python3 calibrate.py

    # Step 2 – apply correction
    python3 correct_bh.py

Output
------
    sino_corrected.pj   – BH-corrected sinogram (same format as input)
"""

import numpy as np
from pj_io import read_pj, write_pj

# ── Config ────────────────────────────────────────────────────────────────
NVIEW      = 360
NDET       = 512
BH_FILE    = "sino_BH.pj"
CAL_FILE   = "calibration.npz"
OUT_FILE   = "sino_corrected.pj"


def load_calibration(cal_file=CAL_FILE):
    """
    Load polynomial coefficients saved by calibrate.py.

    Returns
    -------
    poly_fn : np.poly1d
        Callable that maps p_BH → p_corrected.
    meta : dict
    """
    cal = np.load(cal_file)
    coeffs = cal["coeffs"]          # increasing-power order [c0, c1, …]
    degree = int(cal["degree"])
    r2     = float(cal["r2"])
    rmse   = float(cal["rmse"])

    # np.poly1d expects *decreasing*-power order
    poly_fn = np.poly1d(coeffs[::-1])

    print(f"Loaded calibration from {cal_file!r}")
    print(f"  Degree : {degree}")
    print(f"  R²     : {r2:.6f}")
    print(f"  RMSE   : {rmse:.6f}")
    print(f"  Coeffs : {coeffs}")

    return poly_fn, {"degree": degree, "r2": r2, "rmse": rmse}


def apply_correction(bh_file=BH_FILE,
                     cal_file=CAL_FILE,
                     out_file=OUT_FILE):
    """
    Apply the polynomial correction and write the result.

    Returns
    -------
    sino_corrected : np.ndarray, shape (NVIEW, NDET), float32
    metrics : dict  – pixel-level improvement stats vs ideal (if available)
    """
    print("=== Stage 1 BH Correction ===\n")

    # ── Load calibration ─────────────────────────────────────────────────
    poly_fn, cal_meta = load_calibration(cal_file)

    # ── Load BH sinogram ─────────────────────────────────────────────────
    print(f"\nLoading BH sinogram : {bh_file}")
    sino_bh, bh_meta = read_pj(bh_file, nview=NVIEW, ndet=NDET)

    # ── Apply polynomial correction pixel-wise ────────────────────────────
    print("\nApplying polynomial correction …")
    sino_corrected = poly_fn(sino_bh).astype(np.float32)

    # Clip to valid range – extrapolation can produce slight over/undershoot
    sino_corrected = np.clip(sino_corrected, 0.0, 1.0)

    print(f"Corrected sinogram : min={sino_corrected.min():.6f}  "
          f"max={sino_corrected.max():.6f}")

    # ── Improvement metrics vs ideal (optional) ───────────────────────────
    metrics = {}
    try:
        from pj_io import read_pj as _r
        print("\nLoading ideal sinogram for comparison …")
        sino_ideal, _ = _r("sino_ideal.pj", nview=NVIEW, ndet=NDET)

        err_bh   = sino_bh        - sino_ideal
        err_corr = sino_corrected - sino_ideal

        rmse_bh   = float(np.sqrt(np.mean(err_bh   ** 2)))
        rmse_corr = float(np.sqrt(np.mean(err_corr ** 2)))
        improvement = (1.0 - rmse_corr / rmse_bh) * 100.0

        mae_bh   = float(np.mean(np.abs(err_bh)))
        mae_corr = float(np.mean(np.abs(err_corr)))

        print("\n── Correction quality ──────────────────────────────────")
        print(f"  RMSE  before : {rmse_bh:.6f}")
        print(f"  RMSE  after  : {rmse_corr:.6f}")
        print(f"  Improvement  : {improvement:.2f} %")
        print(f"  MAE   before : {mae_bh:.6f}")
        print(f"  MAE   after  : {mae_corr:.6f}")
        print("────────────────────────────────────────────────────────")

        metrics = {
            "rmse_bh": rmse_bh, "rmse_corrected": rmse_corr,
            "mae_bh": mae_bh,   "mae_corrected": mae_corr,
            "improvement_pct": improvement,
        }
    except FileNotFoundError:
        print("(sino_ideal.pj not found – skipping comparison metrics)")

    # ── Write output ──────────────────────────────────────────────────────
    write_pj(out_file, sino_corrected)
    print(f"\nCorrected sinogram saved → {out_file}")

    return sino_corrected, metrics


if __name__ == "__main__":
    apply_correction()
