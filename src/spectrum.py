"""
spectrum.py — X-ray spectrum and water attenuation modelling.
"""

import numpy as np
from scipy.integrate import trapezoid


class XRaySpectrum:
    """
    Models a polychromatic X-ray tube spectrum and the resulting
    water attenuation coefficients.

    Parameters
    ----------
    kVp : float
        Peak tube voltage in kV (default 80).
    E_min : float
        Minimum energy bin in keV (default 20).
    n_bins : int
        Number of energy bins (default 1000).
    mu_scale : float
        Scale factor for the water attenuation model (default 0.2).
    mu_exp : float
        Power-law exponent for the attenuation model (default -2.7).
    mu_ref_energy : float
        Reference energy for the attenuation model in keV (default 50).
    """

    def __init__(
        self,
        kVp: float = 80,
        E_min: float = 20,
        n_bins: int = 1000,
        mu_scale: float = 0.2,
        mu_exp: float = -2.7,
        mu_ref_energy: float = 50,
    ):
        self.kVp = kVp
        self.E_min = E_min
        self.n_bins = n_bins
        self.mu_scale = mu_scale
        self.mu_exp = mu_exp
        self.mu_ref_energy = mu_ref_energy

        self.E: np.ndarray = np.array([])
        self.S: np.ndarray = np.array([])
        self.mu_water: np.ndarray = np.array([])
        self.E_mean: float = 0.0
        self.mu_mean: float = 0.0

        self._build()

    # ── private ───────────────────────────────────────────────────────────────

    def _build(self) -> None:
        """Compute spectrum and attenuation arrays."""
        self.E = np.linspace(self.E_min, self.kVp, self.n_bins)

        # Simple tungsten-anode approximation
        S = (self.E / self.kVp) * np.exp(-self.E / 30)
        self.S = S / trapezoid(S, self.E)

        self.E_mean = float(trapezoid(self.S * self.E, self.E))

        # Power-law water attenuation
        self.mu_water = np.clip(
            self.mu_scale * (self.E / self.mu_ref_energy) ** self.mu_exp,
            0.02,
            10.0,
        )
        self.mu_mean = float(np.interp(self.E_mean, self.E, self.mu_water))

    # ── public ────────────────────────────────────────────────────────────────

    def summary(self) -> None:
        print(f"    Mean photon energy  : {self.E_mean:.1f} keV")
        print(f"    mu_water @ mean E   : {self.mu_mean:.4f} cm^-1")

    def polychromatic_projection(self, thickness: np.ndarray) -> np.ndarray:
        """
        Compute the polychromatic (beam-hardened) projection value for an
        array of material thicknesses (cm).

        Returns -log(<exp(-mu*t)>_S) for each thickness value.
        """
        t = np.asarray(thickness)
        result = np.empty_like(t, dtype=float)
        for idx in np.ndindex(t.shape):
            integral = trapezoid(
                self.S * np.exp(-self.mu_water * t[idx]), self.E
            )
            result[idx] = -np.log(max(integral, 1e-12))
        return result