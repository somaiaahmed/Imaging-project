"""
plotter.py — Stage 2 summary figure.

Class
-----
Stage2Plotter   Generates and saves the 3-row, 4-column summary plot.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from spectrum import XRaySpectrum
from lut import EmpiricalLUT, PhysicsLUT


class Stage2Plotter:
    """
    Builds the full Stage 2 summary figure.

    Parameters
    ----------
    spectrum : XRaySpectrum
    physics_lut : PhysicsLUT
    empirical_lut : EmpiricalLUT
    output_path : str
        File path for the saved PNG (default 'stage2_summary.png').
    dpi : int
        Resolution of saved figure (default 150).
    """

    def __init__(
        self,
        spectrum: XRaySpectrum,
        physics_lut: PhysicsLUT,
        empirical_lut: EmpiricalLUT,
        output_path: str = "stage2_summary.png",
        dpi: int = 150,
    ):
        self.spectrum      = spectrum
        self.physics_lut   = physics_lut
        self.empirical_lut = empirical_lut
        self.output_path   = output_path
        self.dpi           = dpi

    def plot(
        self,
        sinograms: dict[str, tuple[np.ndarray, str, float]],
        images:    dict[str, tuple[np.ndarray, str, float]],
        sino_bh:   np.ndarray,
        sino_stage1: np.ndarray,
        sino_lut_combined: np.ndarray,
        sino_ideal: np.ndarray,
        recon_ideal: np.ndarray,
    ) -> None:
        """
        Generate and save the summary figure.

        Parameters
        ----------
        sinograms : dict  name -> (array, title, rmse_value)
        images    : dict  name -> (array, title, rmse_value)
        sino_bh, sino_stage1, sino_lut_combined, sino_ideal : row-profile arrays
        recon_ideal : used for vmin/vmax normalisation
        """
        fig = plt.figure(figsize=(20, 14))
        fig.suptitle(
            "Stage 2 — Spectrum LUT Correction + Image Enhancement",
            fontsize=15, fontweight="bold",
        )

        self._plot_sinograms(fig, sinograms)
        self._plot_images(fig, images, recon_ideal)
        self._plot_spectrum(fig)
        self._plot_lut_curves(fig)
        self._plot_row_profile(fig, sino_bh, sino_stage1, sino_lut_combined, sino_ideal)
        self._plot_rmse_bars(fig, images)

        plt.tight_layout()
        plt.savefig(self.output_path, dpi=self.dpi, bbox_inches="tight")
        plt.show()
        print(f"    Saved -> {self.output_path}")

    # ── private helpers ───────────────────────────────────────────────────────

    def _plot_sinograms(self, fig, sinograms):
        for idx, (sino, title, _) in enumerate(sinograms.values()):
            ax = fig.add_subplot(3, 4, idx + 1)
            ax.imshow(sino, cmap="gray", aspect="auto")
            ax.set_title(title, fontsize=9)
            ax.axis("off")

    def _plot_images(self, fig, images, recon_ideal):
        vmin, vmax = recon_ideal.min(), recon_ideal.max()
        for idx, (recon, title, _) in enumerate(images.values()):
            ax = fig.add_subplot(3, 4, idx + 5)
            ax.imshow(recon, cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=9)
            ax.axis("off")

    def _plot_spectrum(self, fig):
        ax = fig.add_subplot(3, 4, 9)
        ax.plot(self.spectrum.E, self.spectrum.S, color="blue", linewidth=2)
        ax.axvline(
            self.spectrum.E_mean, color="red", linestyle="--",
            label=f"Mean={self.spectrum.E_mean:.1f}keV",
        )
        ax.set_title("80kVp Spectrum S(E)")
        ax.set_xlabel("Energy (keV)")
        ax.set_ylabel("Intensity")
        ax.legend(fontsize=8)
        ax.grid(True)

    def _plot_lut_curves(self, fig):
        ax = fig.add_subplot(3, 4, 10)
        ax.plot(
            self.physics_lut.p_poly_norm,
            self.physics_lut.p_ideal_norm,
            color="green", linewidth=2, label="Physics LUT",
        )
        ax.plot(
            self.empirical_lut.bh_centers,
            self.empirical_lut.ideal_means,
            color="orange", linewidth=2, label="Empirical LUT", alpha=0.8,
        )
        ax.plot([0, 1], [0, 1], "r--", linewidth=1, label="Ideal (no BH)")
        ax.set_title("LUT: BH Measured -> Corrected")
        ax.set_xlabel("BH projection value")
        ax.set_ylabel("Corrected value")
        ax.legend(fontsize=8)
        ax.grid(True)

    def _plot_row_profile(self, fig, sino_bh, sino_stage1, sino_lut_combined, sino_ideal):
        ax  = fig.add_subplot(3, 4, 11)
        mid = sino_bh.shape[0] // 2
        ax.plot(sino_bh[mid],           color="red",   label="Original BH",  alpha=0.7)
        ax.plot(sino_stage1[mid],       color="blue",  label="Stage 1",      alpha=0.7)
        ax.plot(sino_lut_combined[mid], color="green", label="Stage 2 LUT",  linewidth=2)
        ax.plot(sino_ideal[mid],        color="black", label="Ideal",        linestyle="--")
        ax.set_title("Sinogram Row Profile")
        ax.set_xlabel("Detector position")
        ax.set_ylabel("Projection value")
        ax.legend(fontsize=8)
        ax.grid(True)

    def _plot_rmse_bars(self, fig, images):
        ax     = fig.add_subplot(3, 4, 12)
        labels = [title for _, title, _ in images.values()]
        values = [rmse  for _, _,     rmse in images.values()]
        colors = ["red", "blue", "green", "black"][: len(values)]
        bars   = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.8)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.01,
                f"{val:.5f}", ha="center", va="bottom", fontsize=7,
            )
        ax.set_title("Image RMSE (lower=better)")
        ax.set_ylabel("RMSE")
        ax.grid(True, axis="y", alpha=0.5)
