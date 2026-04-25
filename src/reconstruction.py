"""
reconstruction.py — Filtered Back-Projection (FBP) reconstructor and
                    post-processing corrections.

Classes
-------
FBPReconstructor   Ramp-filtered back-projection using scipy FFT.
CuppingCorrector   Bias-field cupping correction via uniform filtering.
Evaluator          RMSE metrics for sinogram and image quality.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter
from scipy.fft import fft, ifft, fftfreq


# ── FBP Reconstructor ──────────────────────────────────────────────────────────

class FBPReconstructor:
    """
    Filtered Back-Projection reconstructor (ramp filter, scipy FFT only).

    Usage
    -----
    recon = FBPReconstructor()
    image = recon.reconstruct(sinogram)
    """

    def reconstruct(self, sino: np.ndarray) -> np.ndarray:
        """
        Reconstruct a square image from a (views × detectors) sinogram.

        Parameters
        ----------
        sino : np.ndarray, shape (n_views, n_detectors)

        Returns
        -------
        image : np.ndarray, shape (n_detectors, n_detectors)
        """
        nviews, ndet = sino.shape

        # Ramp filter in frequency domain
        freqs    = fftfreq(ndet)
        ramp     = 2 * np.abs(freqs)
        filtered = np.real(
            ifft(fft(sino, axis=1) * ramp[np.newaxis, :], axis=1)
        )

        # Back-projection
        angles = np.linspace(0, np.pi, nviews, endpoint=False)
        image  = np.zeros((ndet, ndet), dtype=np.float64)
        center = ndet / 2.0
        x      = np.arange(ndet) - center
        xx, yy = np.meshgrid(x, x)

        for i, theta in enumerate(angles):
            t     = xx * np.cos(theta) + yy * np.sin(theta)
            t_idx = np.clip(t + center, 0, ndet - 1)
            t_lo  = np.floor(t_idx).astype(int)
            t_hi  = np.minimum(t_lo + 1, ndet - 1)
            frac  = t_idx - t_lo
            image += (1 - frac) * filtered[i][t_lo] + frac * filtered[i][t_hi]

        image *= np.pi / (2 * nviews)
        return image

    def reconstruct_many(self, **sinograms: np.ndarray) -> dict[str, np.ndarray]:
        """
        Reconstruct multiple sinograms at once.

        Parameters
        ----------
        **sinograms : keyword sinogram arrays

        Returns
        -------
        dict mapping the same keyword names to reconstructed images.

        Example
        -------
        results = recon.reconstruct_many(ideal=sino_ideal, bh=sino_bh)
        """
        results = {}
        for name, sino in sinograms.items():
            print(f"    Reconstructing '{name}' ...")
            img = self.reconstruct(sino)
            print(f"        range [{img.min():.5f}, {img.max():.5f}]")
            results[name] = img
        return results


# ── Cupping Corrector ──────────────────────────────────────────────────────────

class CuppingCorrector:
    """
    Removes low-frequency cupping artefacts via uniform-filter bias estimation.

    Parameters
    ----------
    strength : float
        Fraction of estimated bias to subtract (0–1). Default 0.3.
    filter_fraction : float
        Uniform filter size as a fraction of image width. Default 0.2.
    """

    def __init__(self, strength: float = 0.3, filter_fraction: float = 0.2):
        self.strength = strength
        self.filter_fraction = filter_fraction

    def correct(self, image: np.ndarray) -> np.ndarray:
        """
        Apply cupping correction to a reconstructed image.

        Parameters
        ----------
        image : np.ndarray, shape (H, W)

        Returns
        -------
        corrected : np.ndarray, same shape
        """
        size      = max(1, int(image.shape[0] * self.filter_fraction))
        smoothed  = uniform_filter(image, size=size)
        bias      = smoothed - smoothed.mean()
        return image - bias * self.strength


# ── Evaluator ──────────────────────────────────────────────────────────────────

class Evaluator:
    """
    Computes RMSE metrics between corrected and reference data.

    Parameters
    ----------
    reference_sino : np.ndarray
        Ground-truth sinogram.
    reference_image : np.ndarray
        Ground-truth reconstructed image.
    """

    def __init__(
        self,
        reference_sino: np.ndarray,
        reference_image: np.ndarray,
    ):
        self.reference_sino  = reference_sino
        self.reference_image = reference_image

    # ── static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def rmse(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sqrt(np.mean((a - b) ** 2)))

    @staticmethod
    def center_rmse(a: np.ndarray, b: np.ndarray) -> float:
        """RMSE over the central third of the image (avoids edge artefacts)."""
        cx, cy = a.shape[0] // 2, a.shape[1] // 2
        r      = min(a.shape) // 3
        return float(np.sqrt(np.mean(
            (a[cx - r:cx + r, cy - r:cy + r] - b[cx - r:cx + r, cy - r:cy + r]) ** 2
        )))

    # ── public ────────────────────────────────────────────────────────────────

    def evaluate_sinograms(self, **sinograms: np.ndarray) -> dict[str, float]:
        """
        Return sinogram RMSE vs reference for each supplied sinogram.

        Example
        -------
        metrics = ev.evaluate_sinograms(bh=sino_bh, stage1=sino_stage1)
        """
        return {name: self.rmse(sino, self.reference_sino)
                for name, sino in sinograms.items()}

    def evaluate_images(self, **images: np.ndarray) -> dict[str, float]:
        """
        Return centre-cropped image RMSE vs reference for each supplied image.
        """
        return {name: self.center_rmse(img, self.reference_image)
                for name, img in images.items()}

    def report(
        self,
        sino_metrics: dict[str, float],
        image_metrics: dict[str, float],
        baseline_key: str = "bh",
        stage1_key: str   = "stage1",
        final_key: str    = "final",
    ) -> None:
        print("\n    === Sinogram RMSE ===")
        for name, val in sino_metrics.items():
            print(f"    {name:20s}: {val:.6f}")

        print("\n    === IMAGE RMSE (full precision) ===")
        for name, val in image_metrics.items():
            print(f"    {name:20s}: {val:.10f}")

        if all(k in image_metrics for k in (baseline_key, final_key)):
            improv = (1 - image_metrics[final_key] / image_metrics[baseline_key]) * 100
            print(f"\n    Improvement vs {baseline_key}: {improv:.1f}%")

        if all(k in image_metrics for k in (stage1_key, final_key)):
            improv2 = (1 - image_metrics[final_key] / image_metrics[stage1_key]) * 100
            print(f"    Improvement vs {stage1_key}: {improv2:.1f}%")
