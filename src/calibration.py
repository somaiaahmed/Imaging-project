"""
calibration.py
==============
Stage 1 beam-hardening polynomial calibration and correction.

Classes
-------
BHCalibrator    Fits a polynomial p_ideal = f(p_BH) from paired sinograms.
BHCorrector     Loads a saved calibration and applies it to a new sinogram.
"""

from __future__ import annotations

import numpy as np


# ── Calibrator ─────────────────────────────────────────────────────────────────

class BHCalibrator:
    """
    Fits a polynomial correction curve:
        p_ideal ≈ c0 + c1·p_BH + c2·p_BH² + … + cn·p_BH^n

    Parameters
    ----------
    degree : int
        Polynomial degree (default 3).
    subsample : int
        Use every N-th pixel for speed (default 4).
    min_signal : float
        Ignore pixels below this threshold in both sinograms (default 0.01).
    """

    def __init__(
        self,
        degree: int = 3,
        subsample: int = 4,
        min_signal: float = 0.01,
    ):
        self.degree     = degree
        self.subsample  = subsample
        self.min_signal = min_signal

        self.coeffs: np.ndarray | None = None   # increasing-power order
        self.poly_fn: np.poly1d | None = None
        self.diagnostics: dict = {}

    # ── public ────────────────────────────────────────────────────────────────

    def fit(self, sino_bh: np.ndarray, sino_ideal: np.ndarray) -> "BHCalibrator":
        """
        Fit the polynomial and store coefficients.

        Parameters
        ----------
        sino_bh : np.ndarray   Beam-hardened sinogram.
        sino_ideal : np.ndarray  Ground-truth sinogram (same shape).

        Returns
        -------
        self  (fluent interface)
        """
        x_all = sino_bh.ravel()[:: self.subsample]
        y_all = sino_ideal.ravel()[:: self.subsample]

        mask = (x_all > self.min_signal) & (y_all > self.min_signal)
        x, y = x_all[mask], y_all[mask]

        print(f"    Calibration samples : {x.size:,}  (subsample={self.subsample})")

        # np.polyfit → decreasing power; flip to increasing for clarity
        poly_dec   = np.polyfit(x, y, deg=self.degree)
        self.poly_fn = np.poly1d(poly_dec)
        self.coeffs  = poly_dec[::-1]

        y_pred    = self.poly_fn(x)
        residuals = y - y_pred
        ss_res    = float(np.sum(residuals ** 2))
        ss_tot    = float(np.sum((y - y.mean()) ** 2))

        self.diagnostics = {
            "r2":      1.0 - ss_res / ss_tot,
            "rmse":    float(np.sqrt(np.mean(residuals ** 2))),
            "max_err": float(np.max(np.abs(residuals))),
            "degree":  self.degree,
            "n_samples": x.size,
        }
        self._print_diagnostics()
        return self

    def save(self, path: str = "calibration.npz") -> None:
        """Persist coefficients and diagnostics to an .npz file."""
        if self.coeffs is None:
            raise RuntimeError("Call fit() before save().")
        np.savez(
            path,
            coeffs = self.coeffs,
            degree = np.array(self.degree),
            r2     = np.array(self.diagnostics["r2"]),
            rmse   = np.array(self.diagnostics["rmse"]),
        )
        print(f"    Calibration saved → {path}")

    # ── private ───────────────────────────────────────────────────────────────

    def _print_diagnostics(self) -> None:
        d = self.diagnostics
        print(f"\n    ── Fit quality (degree={self.degree}) ──")
        print(f"    R²       : {d['r2']:.6f}")
        print(f"    RMSE     : {d['rmse']:.6f}")
        print(f"    Max |err|: {d['max_err']:.6f}")
        print(f"\n    ── Coefficients (c0 + c1·x + …) ──")
        for i, c in enumerate(self.coeffs):
            print(f"    c{i} = {c:+.8f}")


# ── Corrector ──────────────────────────────────────────────────────────────────

class BHCorrector:
    """
    Loads a saved polynomial calibration and applies it to a sinogram.

    Parameters
    ----------
    cal_path : str
        Path to calibration.npz produced by BHCalibrator.save().
    clip_range : tuple[float, float]
        Clip corrected values to this range (default (0.0, 1.0)).
    """

    def __init__(
        self,
        cal_path: str = "calibration.npz",
        clip_range: tuple[float, float] = (0.0, 1.0),
    ):
        self.cal_path   = cal_path
        self.clip_range = clip_range

        self.poly_fn: np.poly1d | None = None
        self.meta: dict = {}
        self._load()

    def _load(self) -> None:
        cal = np.load(self.cal_path)
        coeffs = cal["coeffs"]              # increasing-power
        self.poly_fn = np.poly1d(coeffs[::-1])
        self.meta = {
            "degree": int(cal["degree"]),
            "r2":     float(cal["r2"]),
            "rmse":   float(cal["rmse"]),
            "coeffs": coeffs,
        }
        print(f"    Calibration loaded from {self.cal_path!r}")
        print(f"    Degree : {self.meta['degree']}")
        print(f"    R²     : {self.meta['r2']:.6f}")
        print(f"    RMSE   : {self.meta['rmse']:.6f}")

    def correct(self, sino_bh: np.ndarray) -> np.ndarray:
        """
        Apply the polynomial correction.

        Parameters
        ----------
        sino_bh : np.ndarray

        Returns
        -------
        corrected : np.ndarray, float32, clipped to clip_range
        """
        corrected = self.poly_fn(sino_bh).astype(np.float32)
        corrected = np.clip(corrected, *self.clip_range)
        print(
            f"    Corrected range: "
            f"[{corrected.min():.6f}, {corrected.max():.6f}]"
        )
        return corrected

    def evaluate(
        self,
        sino_bh: np.ndarray,
        sino_corrected: np.ndarray,
        sino_ideal: np.ndarray,
    ) -> dict[str, float]:
        """
        Compute before/after RMSE and MAE vs ground truth.

        Returns
        -------
        dict with keys: rmse_bh, rmse_corrected, mae_bh, mae_corrected,
                        improvement_pct
        """
        def _rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
        def _mae(a, b):  return float(np.mean(np.abs(a - b)))

        rmse_bh   = _rmse(sino_bh,        sino_ideal)
        rmse_corr = _rmse(sino_corrected,  sino_ideal)
        mae_bh    = _mae(sino_bh,          sino_ideal)
        mae_corr  = _mae(sino_corrected,   sino_ideal)
        improv    = (1.0 - rmse_corr / rmse_bh) * 100.0

        print("\n    ── Correction quality ──")
        print(f"    RMSE  before : {rmse_bh:.6f}")
        print(f"    RMSE  after  : {rmse_corr:.6f}")
        print(f"    Improvement  : {improv:.2f} %")
        print(f"    MAE   before : {mae_bh:.6f}")
        print(f"    MAE   after  : {mae_corr:.6f}")

        return {
            "rmse_bh": rmse_bh, "rmse_corrected": rmse_corr,
            "mae_bh":  mae_bh,  "mae_corrected":  mae_corr,
            "improvement_pct": improv,
        }