"""
plotter.py
==========
Visualisation helpers for both pipeline stages.

Classes
-------
Stage1Plotter   4-panel sinogram correction summary (Stage 1).
Stage2Plotter   3-row 12-panel LUT + reconstruction summary (Stage 2).
SinogramViewer  Quick side-by-side viewer for arbitrary sinogram arrays.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from spectrum import XRaySpectrum
from lut import EmpiricalLUT, PhysicsLUT


# ── helpers ────────────────────────────────────────────────────────────────────

def _pct_clim(arr: np.ndarray, lo: float = 1, hi: float = 99):
    return float(np.percentile(arr, lo)), float(np.percentile(arr, hi))


# ── Stage 1 Plotter ────────────────────────────────────────────────────────────

class Stage1Plotter:
    """
    Generates the Stage 1 correction summary figure:

      Row 0 : ideal | BH-corrupted | corrected | calibration curve
      Row 1 : error map (BH) | error map (corrected) | detector profile

    Parameters
    ----------
    output_path : str
    dpi : int
    """

    def __init__(
        self,
        output_path: str = "correction_summary.png",
        dpi: int = 150,
    ):
        self.output_path = output_path
        self.dpi         = dpi

    def plot(
        self,
        sinos: dict[str, np.ndarray],
        calibration: dict | None = None,
        n_views: int = 360,
    ) -> None:
        """
        Parameters
        ----------
        sinos : dict with any of keys 'ideal', 'bh', 'corrected'
        calibration : dict with keys 'coeffs', 'degree', 'r2'  (optional)
        n_views : used to pick the midpoint profile angle
        """
        fig = plt.figure(figsize=(18, 10))
        fig.suptitle(
            "Stage 1 Beam-Hardening Polynomial Correction",
            fontsize=14, y=1.01,
        )
        gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

        self._row0_sinograms(fig, gs, sinos)
        self._row0_calibration(fig, gs, sinos, calibration)
        self._row1_error_maps(fig, gs, sinos)
        self._row1_profile(fig, gs, sinos, n_views)

        plt.savefig(self.output_path, dpi=self.dpi, bbox_inches="tight")
        print(f"    Figure saved → {self.output_path}")
        plt.show()

    # ── private ───────────────────────────────────────────────────────────────

    def _row0_sinograms(self, fig, gs, sinos):
        ref  = sinos.get("ideal", next(iter(sinos.values())))
        vmin, vmax = _pct_clim(ref)
        panels = [("ideal",     "Ideal (ground truth)"),
                  ("bh",        "BH-corrupted"),
                  ("corrected", "Stage-1 corrected")]

        for col, (key, title) in enumerate(panels):
            ax = fig.add_subplot(gs[0, col])
            if key not in sinos:
                ax.set_visible(False)
                continue
            im = ax.imshow(sinos[key], cmap="gray", aspect="auto",
                           vmin=vmin, vmax=vmax, origin="upper")
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Detector index", fontsize=8)
            ax.set_ylabel("Projection angle", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def _row0_calibration(self, fig, gs, sinos, calibration):
        ax = fig.add_subplot(gs[0, 3])
        if calibration and "bh" in sinos and "ideal" in sinos:
            coeffs = calibration["coeffs"]
            degree = calibration["degree"]
            r2     = calibration["r2"]

            x_all = sinos["bh"].ravel()[::20]
            y_all = sinos["ideal"].ravel()[::20]
            mask  = (x_all > 0.01) & (y_all > 0.01)
            ax.scatter(x_all[mask], y_all[mask],
                       s=0.3, alpha=0.15, color="steelblue", label="data")

            t = np.linspace(0, 1, 300)
            poly_fn = np.poly1d(coeffs[::-1])
            ax.plot(t, poly_fn(t), "r-", lw=2,
                    label=f"poly deg={degree}\nR²={r2:.4f}")
            ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="identity")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_xlabel("p_BH  (corrupted)", fontsize=8)
            ax.set_ylabel("p_ideal  (target)", fontsize=8)
            ax.legend(fontsize=7, markerscale=5)
        else:
            ax.text(0.5, 0.5, "calibration.npz\nnot found",
                    ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Calibration curve", fontsize=10)

    def _row1_error_maps(self, fig, gs, sinos):
        if "ideal" not in sinos:
            return
        for col, (key, title) in enumerate([
            ("bh",        "Error: BH − Ideal"),
            ("corrected", "Error: Corrected − Ideal"),
        ]):
            ax = fig.add_subplot(gs[1, col])
            if key not in sinos:
                ax.set_visible(False)
                continue
            diff = sinos[key] - sinos["ideal"]
            amax = float(np.percentile(np.abs(diff), 99))
            im   = ax.imshow(diff, cmap="RdBu_r", aspect="auto",
                             vmin=-amax, vmax=amax, origin="upper")
            rmse = float(np.sqrt(np.mean(diff ** 2)))
            ax.set_title(title, fontsize=10)
            ax.set_xlabel(f"Detector index   [RMSE={rmse:.5f}]", fontsize=8)
            ax.set_ylabel("Projection angle", fontsize=8)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label("Δ value", fontsize=7)

    def _row1_profile(self, fig, gs, sinos, n_views):
        ax  = fig.add_subplot(gs[1, 2:])
        mid = n_views // 2
        styles = {
            "ideal":     ("k-",  1.5, "Ideal"),
            "bh":        ("r--", 1.2, "BH-corrupted"),
            "corrected": ("g-",  1.2, "Corrected"),
        }
        for key, (style, lw, label) in styles.items():
            if key in sinos:
                ax.plot(sinos[key][mid], style, lw=lw, label=label, alpha=0.85)
        ax.set_title(f"Detector profile at angle {mid}°", fontsize=10)
        ax.set_xlabel("Detector index", fontsize=8)
        ax.set_ylabel("Projection value", fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)


# ── Stage 2 Plotter ────────────────────────────────────────────────────────────

class Stage2Plotter:
    """
    3-row, 12-panel Stage 2 summary figure.

    Parameters
    ----------
    spectrum : XRaySpectrum
    physics_lut : PhysicsLUT
    empirical_lut : EmpiricalLUT
    output_path : str
    dpi : int
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
        sino_bh:           np.ndarray,
        sino_stage1:       np.ndarray,
        sino_lut_combined: np.ndarray,
        sino_ideal:        np.ndarray,
        recon_ideal:       np.ndarray,
    ) -> None:
        """
        Parameters
        ----------
        sinograms / images : dicts  name -> (array, title, rmse)
        sino_* arrays       : used for the row-profile panel
        recon_ideal         : sets the shared colour limits for image panels
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
        print(f"    Saved → {self.output_path}")

    # ── private ───────────────────────────────────────────────────────────────

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
        ax.axvline(self.spectrum.E_mean, color="red", linestyle="--",
                   label=f"Mean={self.spectrum.E_mean:.1f}keV")
        ax.set_title("80kVp Spectrum S(E)")
        ax.set_xlabel("Energy (keV)")
        ax.set_ylabel("Intensity")
        ax.legend(fontsize=8)
        ax.grid(True)

    def _plot_lut_curves(self, fig):
        ax = fig.add_subplot(3, 4, 10)
        ax.plot(self.physics_lut.p_poly_norm,  self.physics_lut.p_ideal_norm,
                color="green",  linewidth=2, label="Physics LUT")
        ax.plot(self.empirical_lut.bh_centers, self.empirical_lut.ideal_means,
                color="orange", linewidth=2, label="Empirical LUT", alpha=0.8)
        ax.plot([0, 1], [0, 1], "r--", linewidth=1, label="Ideal (no BH)")
        ax.set_title("LUT: BH Measured → Corrected")
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


# ── Sinogram Viewer ────────────────────────────────────────────────────────────

class SinogramViewer:
    """
    Quick side-by-side display of one or more sinograms.
    """

    def show(self, sinograms: dict[str, np.ndarray]) -> None:
        """
        Parameters
        ----------
        sinograms : dict  title -> array
        """
        n   = len(sinograms)
        fig, axes = plt.subplots(1, n, figsize=(7 * n, 6), squeeze=False)

        for ax, (title, sino) in zip(axes[0], sinograms.items()):
            lo, hi = _pct_clim(sino)
            im = ax.imshow(sino, cmap="gray", aspect="auto",
                           vmin=lo, vmax=hi, origin="upper")
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("Detector index")
            ax.set_ylabel("Projection angle")
            fig.colorbar(im, ax=ax, label="Projection value")
            print(f"\n── {title} ──")
            print(f"  Shape : {sino.shape}")
            print(f"  Min   : {sino.min():.6f}  Max: {sino.max():.6f}")

        fig.tight_layout()
        plt.show()