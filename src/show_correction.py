"""
show_correction.py
==================
4-panel visualisation of the Stage 1 BH correction pipeline:

  Panel 1 – Ideal sinogram (ground truth)
  Panel 2 – BH-corrupted sinogram
  Panel 3 – Stage-1 corrected sinogram
  Panel 4 – Calibration curve + error maps (difference images)

Usage
-----
    python3 show_correction.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pj_io import read_pj

NVIEW = 360
NDET  = 512


def percentile_clim(arr, lo=1, hi=99):
    return np.percentile(arr, lo), np.percentile(arr, hi)


def load_all():
    sinos = {}
    for key, fname in [("ideal",     "sino_ideal.pj"),
                       ("bh",        "sino_BH.pj"),
                       ("corrected", "sino_corrected.pj")]:
        try:
            sino, meta = read_pj(fname, nview=NVIEW, ndet=NDET)
            sinos[key] = sino
            print(f"Loaded {fname}")
        except FileNotFoundError:
            print(f"WARNING: {fname} not found – skipping.")
    return sinos


def load_calibration():
    try:
        cal = np.load("calibration.npz")
        return cal["coeffs"], int(cal["degree"]), float(cal["r2"])
    except FileNotFoundError:
        return None, None, None


def plot(sinos, coeffs, degree, r2):
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("Stage 1 Beam-Hardening Polynomial Correction", fontsize=14, y=1.01)

    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

    # ── Row 0: sinograms ─────────────────────────────────────────────────
    labels = [("ideal", "Ideal (ground truth)"),
              ("bh",    "BH-corrupted"),
              ("corrected", "Stage-1 corrected")]

    vmin, vmax = percentile_clim(sinos.get("ideal", next(iter(sinos.values()))))

    sino_axes = []
    for col, (key, title) in enumerate(labels):
        ax = fig.add_subplot(gs[0, col])
        sino_axes.append(ax)
        if key not in sinos:
            ax.set_visible(False)
            continue
        im = ax.imshow(sinos[key], cmap="gray", aspect="auto",
                       vmin=vmin, vmax=vmax, origin="upper")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Detector index", fontsize=8)
        ax.set_ylabel("Projection angle", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ── Row 0, col 3: calibration curve ──────────────────────────────────
    ax_cal = fig.add_subplot(gs[0, 3])
    if coeffs is not None and "bh" in sinos and "ideal" in sinos:
        # Scatter (subsampled)
        x_all = sinos["bh"].ravel()[::20]
        y_all = sinos["ideal"].ravel()[::20]
        mask  = (x_all > 0.01) & (y_all > 0.01)
        ax_cal.scatter(x_all[mask], y_all[mask],
                       s=0.3, alpha=0.15, color="steelblue", label="data")

        # Polynomial curve
        t = np.linspace(0, 1, 300)
        poly_fn = np.poly1d(coeffs[::-1])
        ax_cal.plot(t, poly_fn(t), "r-", lw=2,
                    label=f"poly deg={degree}\nR²={r2:.4f}")

        # Identity line
        ax_cal.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="identity")

        ax_cal.set_xlim(0, 1); ax_cal.set_ylim(0, 1)
        ax_cal.set_xlabel("p_BH  (corrupted)", fontsize=8)
        ax_cal.set_ylabel("p_ideal  (target)", fontsize=8)
        ax_cal.set_title("Calibration curve", fontsize=10)
        ax_cal.legend(fontsize=7, markerscale=5)
    else:
        ax_cal.text(0.5, 0.5, "calibration.npz\nnot found",
                    ha="center", va="center", transform=ax_cal.transAxes)
        ax_cal.set_title("Calibration curve", fontsize=10)

    # ── Row 1: difference images ──────────────────────────────────────────
    diff_pairs = [
        ("bh",        "Error: BH − Ideal"),
        ("corrected", "Error: Corrected − Ideal"),
    ]

    if "ideal" in sinos:
        for col, (key, title) in enumerate(diff_pairs):
            ax = fig.add_subplot(gs[1, col])
            if key not in sinos:
                ax.set_visible(False)
                continue
            diff = sinos[key] - sinos["ideal"]
            amax = np.percentile(np.abs(diff), 99)
            im = ax.imshow(diff, cmap="RdBu_r", aspect="auto",
                           vmin=-amax, vmax=amax, origin="upper")
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Detector index", fontsize=8)
            ax.set_ylabel("Projection angle", fontsize=8)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label("Δ value", fontsize=7)

            # Annotate with RMSE
            rmse = float(np.sqrt(np.mean(diff ** 2)))
            ax.set_xlabel(f"Detector index   [RMSE={rmse:.5f}]", fontsize=8)

    # ── Row 1, col 2-3: profile comparison ───────────────────────────────
    ax_prof = fig.add_subplot(gs[1, 2:])
    if "ideal" in sinos:
        mid_angle = NVIEW // 2
        ax_prof.plot(sinos["ideal"][mid_angle],
                     "k-",  lw=1.5, label="Ideal")
        if "bh" in sinos:
            ax_prof.plot(sinos["bh"][mid_angle],
                         "r--", lw=1.2, label="BH-corrupted", alpha=0.8)
        if "corrected" in sinos:
            ax_prof.plot(sinos["corrected"][mid_angle],
                         "g-",  lw=1.2, label="Corrected", alpha=0.9)
        ax_prof.set_title(f"Detector profile at angle {mid_angle}°", fontsize=10)
        ax_prof.set_xlabel("Detector index", fontsize=8)
        ax_prof.set_ylabel("Projection value", fontsize=8)
        ax_prof.legend(fontsize=8)
        ax_prof.grid(True, alpha=0.3)

    plt.savefig("correction_summary.png", dpi=150, bbox_inches="tight")
    print("\nFigure saved → correction_summary.png")
    plt.show()


if __name__ == "__main__":
    sinos = load_all()
    coeffs, degree, r2 = load_calibration()
    plot(sinos, coeffs, degree, r2)
