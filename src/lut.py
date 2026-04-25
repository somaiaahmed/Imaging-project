"""
lut.py — Beam-hardening look-up-table (LUT) correction strategies.

Classes
-------
BaseLUT          Abstract base — defines the apply() interface.
PhysicsLUT       Physics-derived LUT built from a spectrum model.
EmpiricalLUT     Data-driven LUT built from paired (BH, ideal) sinograms.
BlendedLUT       Weighted combination of two LUTs for regularisation.
"""

from __future__ import annotations

import abc
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid

from spectrum import XRaySpectrum


# ── Abstract base ──────────────────────────────────────────────────────────────

class BaseLUT(abc.ABC):
    """Abstract look-up-table that maps BH projection values to corrected ones."""

    @abc.abstractmethod
    def build(self) -> None:
        """Build internal interpolator."""

    @abc.abstractmethod
    def apply(self, sino: np.ndarray) -> np.ndarray:
        """Apply the LUT to a sinogram, replacing any bad pixels safely."""

    # ── shared utility ────────────────────────────────────────────────────────

    @staticmethod
    def _safe_apply(
        fn, sino: np.ndarray, fallback: np.ndarray
    ) -> np.ndarray:
        out = fn(sino)
        bad = np.isnan(out) | np.isinf(out)
        if bad.any():
            print(f"    Warning: {bad.sum()} bad pixels replaced with fallback")
            out[bad] = fallback[bad]
        return out


# ── Physics LUT ────────────────────────────────────────────────────────────────

class PhysicsLUT(BaseLUT):
    """
    LUT derived from a forward polychromatic model.

    Maps normalised BH projections to the equivalent monoenergetic
    (ideal) projections using the supplied XRaySpectrum.

    Parameters
    ----------
    spectrum : XRaySpectrum
        Pre-built spectrum object.
    t_max : float
        Maximum physical path length in cm (default 8.0).
    n_samples : int
        Number of thickness samples (default 5000).
    """

    def __init__(
        self,
        spectrum: XRaySpectrum,
        t_max: float = 8.0,
        n_samples: int = 5000,
    ):
        self.spectrum = spectrum
        self.t_max = t_max
        self.n_samples = n_samples

        self._interp = None
        self.p_poly_norm: np.ndarray = np.array([])
        self.p_ideal_norm: np.ndarray = np.array([])

        self.build()

    def build(self) -> None:
        t_physical = np.linspace(1e-6, 1.0, self.n_samples) * self.t_max

        p_poly = self.spectrum.polychromatic_projection(t_physical)
        p_ideal = self.spectrum.mu_mean * t_physical

        p_min, p_max = p_poly.min(), p_poly.max()
        p_poly_norm  = (p_poly  - p_min) / (p_max - p_min)
        p_ideal_norm = (p_ideal - p_min) / (p_max - p_min)

        # Enforce monotonicity
        mono = np.append(np.diff(p_poly_norm) > 0, True)
        self.p_poly_norm  = p_poly_norm[mono]
        self.p_ideal_norm = p_ideal_norm[mono]

        self._interp = interp1d(
            self.p_poly_norm, self.p_ideal_norm,
            bounds_error=False,
            fill_value=(self.p_ideal_norm[0], self.p_ideal_norm[-1]),
        )
        print(
            f"    PhysicsLUT built — range: "
            f"{self.p_poly_norm.min():.4f} -> {self.p_poly_norm.max():.4f}"
        )

    def apply(self, sino: np.ndarray) -> np.ndarray:
        return self._safe_apply(self._interp, sino, sino)


# ── Empirical LUT ──────────────────────────────────────────────────────────────

class EmpiricalLUT(BaseLUT):
    """
    Data-driven LUT using a binned-mean mapping from BH to ideal sinograms.

    Parameters
    ----------
    sino_bh : np.ndarray
        Beam-hardened sinogram (views × detectors).
    sino_ideal : np.ndarray
        Ground-truth monoenergetic sinogram (same shape).
    n_bins : int
        Number of histogram bins (default 500).
    """

    def __init__(
        self,
        sino_bh: np.ndarray,
        sino_ideal: np.ndarray,
        n_bins: int = 500,
    ):
        self.sino_bh = sino_bh
        self.sino_ideal = sino_ideal
        self.n_bins = n_bins

        self._interp = None
        self.bh_centers: np.ndarray = np.array([])
        self.ideal_means: np.ndarray = np.array([])

        self.build()

    def build(self) -> None:
        bh_flat    = self.sino_bh.flatten()
        ideal_flat = self.sino_ideal.flatten()

        edges = np.linspace(bh_flat.min(), bh_flat.max(), self.n_bins + 1)
        self.bh_centers = 0.5 * (edges[:-1] + edges[1:])
        ideal_means = np.zeros(self.n_bins)

        for i in range(self.n_bins):
            mask = (bh_flat >= edges[i]) & (bh_flat < edges[i + 1])
            ideal_means[i] = ideal_flat[mask].mean() if mask.sum() > 0 else np.nan

        # Fill any empty bins by interpolation
        valid = ~np.isnan(ideal_means)
        self.ideal_means = np.interp(
            self.bh_centers, self.bh_centers[valid], ideal_means[valid]
        )

        self._interp = interp1d(
            self.bh_centers, self.ideal_means,
            bounds_error=False,
            fill_value=(self.ideal_means[0], self.ideal_means[-1]),
        )

        test_range = self._interp(self.sino_bh)
        print(
            f"    EmpiricalLUT built — output range: "
            f"{test_range.min():.4f} -> {test_range.max():.4f}"
        )

    def apply(self, sino: np.ndarray) -> np.ndarray:
        return self._safe_apply(self._interp, sino, sino)


# ── Blended LUT ────────────────────────────────────────────────────────────────

class BlendedLUT(BaseLUT):
    """
    Weighted blend of two LUTs.

    Parameters
    ----------
    primary : BaseLUT
        Main correction LUT (higher weight).
    secondary : BaseLUT
        Regularising LUT or raw sinogram pass-through.
    alpha : float
        Weight for primary (0–1). secondary gets (1 - alpha).
    """

    def __init__(
        self,
        primary: BaseLUT,
        secondary: BaseLUT,
        alpha: float = 0.9,
    ):
        self.primary = primary
        self.secondary = secondary
        self.alpha = alpha

    def build(self) -> None:
        # Constituent LUTs are already built at construction time.
        pass

    def apply(self, sino: np.ndarray) -> np.ndarray:
        out = self.alpha * self.primary.apply(sino) + (1 - self.alpha) * self.secondary.apply(sino)
        return self._safe_apply(lambda x: x, out, sino)