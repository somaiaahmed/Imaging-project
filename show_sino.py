"""
show_sino.py
============
Visualise one or more .pj sinograms.

Usage
-----
    python3 show_sino.py                       # shows sino_BH.pj only
    python3 show_sino.py sino_ideal.pj sino_BH.pj  # side-by-side
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pj_io import read_pj

NVIEW = 360
NDET  = 512


def show(files):
    n = len(files)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6), squeeze=False)

    for ax, fname in zip(axes[0], files):
        try:
            sino, meta = read_pj(fname, nview=NVIEW, ndet=NDET)
        except Exception as e:
            print(f"Could not read {fname!r}: {e}")
            continue

        # Clip to [1st, 99th] percentile for robust display
        lo, hi = np.percentile(sino, 1), np.percentile(sino, 99)
        im = ax.imshow(
            sino,
            cmap="gray",
            aspect="auto",
            vmin=lo,
            vmax=hi,
            origin="upper",
        )
        ax.set_title(fname, fontsize=11)
        ax.set_xlabel("Detector index")
        ax.set_ylabel("Projection angle")
        fig.colorbar(im, ax=ax, label="Projection value")

        print(f"\n── {fname} ──")
        print(f"  Shape  : {sino.shape}")
        print(f"  Min    : {sino.min():.6f}")
        print(f"  Max    : {sino.max():.6f}")
        print(f"  Offset : {meta['offset']}")
        print(f"  Scale  : {meta['raw_scale']:.6f}")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    files = sys.argv[1:] if len(sys.argv) > 1 else ["sino_BH.pj"]
    show(files)
