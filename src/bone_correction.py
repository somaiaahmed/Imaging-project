from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from scipy.ndimage import binary_fill_holes, binary_erosion
from scipy.fft import fft, ifft, fftfreq
from phantom import ForwardProjector
from reconstruction import FBPReconstructor


# BH strength presets to mimic the artifacts

_BH_STRENGTH: dict[str, float] = {
    "mild":   0.30,
    "strong": 0.55,
    "severe": 0.80,
}


# custom forward projector that matches the FBPReconstructor geometry and interpolation
# the old one from phantom mismatch the geometry so making like two objects overlays on each other

class MatchedForwardProjector:
    """
    Forward projector that is the exact transpose of FBPReconstructor.
    """

    def __init__(self, n_views: int = 360, n_det: int = 512):
        self.n_views = n_views
        self.n_det   = n_det
        self.angles  = np.linspace(0, np.pi, n_views, endpoint=False)

        center   = n_det / 2.0
        x        = np.arange(n_det) - center
        xx, yy   = np.meshgrid(x, x)
        self._xx = xx.ravel()
        self._yy = yy.ravel()

    def project(self, image: np.ndarray) -> np.ndarray:
        """
        forward project image into sinogram (raw units).
        takes image and return the sino
        """
        center   = self.n_det / 2.0
        img_flat = image.ravel().astype(np.float64)
        sino     = np.zeros((self.n_views, self.n_det), dtype=np.float64)

        for i, theta in enumerate(self.angles):
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)

            t     = self._xx * cos_t + self._yy * sin_t
            t_idx = t + center

            t_lo  = np.floor(t_idx).astype(int)
            t_hi  = t_lo + 1
            frac  = t_idx - t_lo

            valid = (t_lo >= 0) & (t_hi < self.n_det)

            np.add.at(sino[i], t_lo[valid], (1.0 - frac[valid]) * img_flat[valid])
            np.add.at(sino[i], t_hi[valid],        frac[valid]  * img_flat[valid])

        return sino


class BoneSegmenter:
    """
    segment bone from a reconstructed CT image using a relative HU threshold (300 HU above soft tissue)
    takes the hu_threshold and erode_px parameters (that remove partial volume boundary voxels)
    returns the mask
    """

    def __init__(self, hu_threshold: float = 700.0, erode_px: int = 3):
        self.hu_threshold = hu_threshold
        self.erode_px     = erode_px
        self._rel         = 1.0 + hu_threshold / 1000.0

    def segment(self, image: np.ndarray) -> np.ndarray:
        positive = image[image > 1e-6 * image.max()]
        if positive.size == 0:
            return np.zeros(image.shape, dtype=bool)

        soft_ref  = np.percentile(positive, 50)
        if soft_ref <= 0:
            return np.zeros(image.shape, dtype=bool)

        mask = image >= soft_ref * self._rel
        mask = binary_fill_holes(mask)

        if self.erode_px > 0:
            struct = np.ones((2 * self.erode_px + 1,) * 2, dtype=bool)
            mask   = binary_erosion(mask, structure=struct)

        return mask.astype(bool)

# the stage three method (loop of segment and correct)
class IterativeBoneCorrector:
    """
    n_views      : int
    n_det        : int
    severity     : str   "mild" | "strong" | "severe"  (default "strong")
    max_iter     : int   default 15
    tol          : float stop when delta_sino < tol  (default 1e-3)
    lam          : float damping factor in (0,1]  (default 0.30)
    hu_threshold : float HU threshold for bone mask  (default 700)
    erode_px     : int   bone mask erosion radius  (default 3)
    verbose      : bool
    """

    def __init__(
        self,
        n_views:      int   = 360,
        n_det:        int   = 512,
        severity:     str   = "strong",
        max_iter:     int   = 15,
        tol:          float = 1e-3,
        lam:          float = 0.30,
        hu_threshold: float = 700.0,
        erode_px:     int   = 3,
        verbose:      bool  = True,
    ):
        self.n_views     = n_views
        self.n_det       = n_det
        self.max_iter    = max_iter
        self.tol         = tol
        self.lam         = lam
        self.verbose     = verbose

        if severity not in _BH_STRENGTH:
            raise ValueError(f"severity must be one of {list(_BH_STRENGTH)}")
        self.bh_strength = _BH_STRENGTH[severity]

        self._projector  = MatchedForwardProjector(n_views=n_views, n_det=n_det)
        self._recon      = FBPReconstructor()
        self._segmenter  = BoneSegmenter(
            hu_threshold=hu_threshold, erode_px=erode_px
        )

        self.history: list[dict] = []
        self._P_full_max: float  = 1.0   # set on first call to correct()

    @staticmethod
    def _rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))

    @staticmethod
    def _rms(a):     return float(np.sqrt(np.mean(a ** 2)))

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def correct(
        self,
        sino_raw:        np.ndarray,
        reference_sino:  np.ndarray | None = None,
        reference_image: np.ndarray | None = None,
    ) -> dict:
        """
        run iterative bone correction
        """
        self.history = []

        self._log("\n=== ITERATIVE BONE CORRECTION ===")
        self._log("  k=0  FBP of raw sinogram ...")
        image_k = self._recon.reconstruct(sino_raw)
        sino_k  = sino_raw.copy()

        self._log("  Calibrating projection scale from full image ...")
        p_full_raw       = self._projector.project(image_k)
        self._P_full_max = float(p_full_raw.max()) if p_full_raw.max() > 0 else 1.0
        self._log(f"  P_full_max = {self._P_full_max:.4f}")

        rms_raw = self._rms(sino_raw)

        self.history.append({
            "k": 0, "image": image_k.copy(), "sino": sino_k.copy(),
            "bone_mask": None, "p_bone": None, "bh_err": None,
            "delta_sino": None,
            "rmse_vs_ref_sino":  (
                self._rmse(sino_raw, reference_sino)
                if reference_sino  is not None else None),
            "rmse_vs_ref_image": (
                self._rmse(image_k, reference_image)
                if reference_image is not None else None),
        })

        if reference_sino is not None:
            self._log(f"    sino  RMSE vs ideal : {self.history[0]['rmse_vs_ref_sino']:.6f}")
        if reference_image is not None:
            self._log(f"    image RMSE vs ideal : {self.history[0]['rmse_vs_ref_image']:.6f}")

        converged = False
        k = 0

        for k in range(1, self.max_iter + 1):
            self._log(f"\n  k={k}")

            # segment bone
            bone_mask = self._segmenter.segment(image_k)
            n_bone    = int(bone_mask.sum())
            self._log(f"    bone voxels : {n_bone:,}")
            if n_bone == 0:
                self._log("    WARNING: no bone detected — stopping")
                break

            # bone-only image
            image_bone = image_k * bone_mask.astype(image_k.dtype)

            # project the bone image
            p_bone_raw = self._projector.project(image_bone)
            # normalize to 0 and 1
            p_bone = (p_bone_raw / self._P_full_max).astype(np.float32)
            # BH error: S_bh = S - bh_str*S^2  =>  missing = bh_str*P_bone^2
            bh_err = self.bh_strength * p_bone ** 2
            # update
            sino_new = np.clip(sino_k + self.lam * bh_err, 0.0, 1.0)
            # reconstruct
            image_new = self._recon.reconstruct(sino_new)
            # convergence metrics
            delta_sino     = self._rmse(sino_new, sino_k) / (rms_raw + 1e-12)
            rmse_ref_sino  = None
            rmse_ref_image = None
            if reference_sino is not None:
                rmse_ref_sino  = self._rmse(sino_new,  reference_sino)
                rmse_ref_image = self._rmse(image_new, reference_image)
                self._log(f"    sino  RMSE vs ideal : {rmse_ref_sino:.6f}")
                self._log(f"    image RMSE vs ideal : {rmse_ref_image:.6f}")
            self._log(f"    delta sino (rel)    : {delta_sino:.4e}  (tol={self.tol:.1e})")

            self.history.append({
                "k": k, "image": image_new.copy(), "sino": sino_new.copy(),
                "bone_mask": bone_mask, "p_bone": p_bone, "bh_err": bh_err,
                "delta_sino": delta_sino,
                "rmse_vs_ref_sino":  rmse_ref_sino,
                "rmse_vs_ref_image": rmse_ref_image,
            })

            sino_k  = sino_new
            image_k = image_new

            # early stopping using RMSe
            if (rmse_ref_image is not None and len(self.history) >= 3
                    and self.history[-2]["rmse_vs_ref_image"] is not None):
                prev = self.history[-2]["rmse_vs_ref_image"]
                if rmse_ref_image > prev * 1.005:
                    self._log(
                        f"\n  Early stop k={k}: image RMSE rose "
                        f"({prev:.6f} -> {rmse_ref_image:.6f}). Rollback to k={k-1}."
                    )
                    best = self.history[-2]
                    sino_k  = best["sino"].copy()
                    image_k = best["image"].copy()
                    converged = True
                    break

            if delta_sino < self.tol:
                self._log(f"\n  Converged at k={k} (delta={delta_sino:.2e})")
                converged = True
                break

        if not converged:
            self._log(f"\n  WARNING: reached max_iter={self.max_iter}")

        return {
            "sino_corrected":  sino_k,
            "image_corrected": image_k,
            "image_initial":   self.history[0]["image"],
            "converged":       converged,
            "n_iter":          k,
            "history":         self.history,
        }

    def print_summary(self) -> None:
        print("\n  +-----+--------------+--------------+--------------+")
        print(  "  |  k  |  delta sino  | sino vs ref  |  img vs ref  |")
        print(  "  +-----+--------------+--------------+--------------+")
        for h in self.history:
            ds = f"{h['delta_sino']:.4e}"        if h["delta_sino"]         is not None else "      -      "
            rs = f"{h['rmse_vs_ref_sino']:.6f}"  if h["rmse_vs_ref_sino"]   is not None else "     -      "
            ri = f"{h['rmse_vs_ref_image']:.6f}" if h["rmse_vs_ref_image"]  is not None else "     -      "
            print(f"  | {h['k']:3d} | {ds:12s} | {rs:12s} | {ri:12s} |")
        print(  "  +-----+--------------+--------------+--------------+")


# ════════════════════════════════════════════════════════════════════════════════
# Plotter
# ════════════════════════════════════════════════════════════════════════════════

class BoneCorrectionPlotter:
    _BG      = "#0d1117"
    _FG      = "#e6edf3"
    _ACCENT  = "#58a6ff"
    _RED     = "#f85149"
    _GREEN   = "#3fb950"
    _YELLOW  = "#d29922"
    _PANEL   = "#161b22"
    _BORDER  = "#30363d"
    _CT_CMAP   = "bone"
    _DIFF_CMAP = "RdBu_r"
    _SINO_CMAP = "inferno"
    _MASK_CMAP = "Greens"

    def __init__(self, output_path: str, dpi: int = 150):
        self.output_path = output_path
        self.dpi         = dpi

    @staticmethod
    def _rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))

    def _panel(self, ax):
        ax.set_facecolor(self._PANEL)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(self._BORDER)

    def _cbar(self, fig, ax, im, label=""):
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.yaxis.set_tick_params(color=self._FG, labelcolor=self._FG, labelsize=6)
        cb.outline.set_edgecolor(self._BORDER)
        if label:
            cb.set_label(label, color=self._FG, fontsize=6.5)

    def plot(
        self,
        sino_bh, sino_corrected, sino_ideal,
        image_bh, image_corrected, image_ideal,
        bone_mask, history,
    ) -> None:

        fig = plt.figure(figsize=(16, 14), facecolor=self._BG)
        fig.subplots_adjust(hspace=0.44, wspace=0.30,
                            left=0.04, right=0.96, top=0.93, bottom=0.05)
        gs = gridspec.GridSpec(4, 3, figure=fig,
                               height_ratios=[1.1, 1.1, 1.1, 1.0])

        n_iter   = max(h["k"] for h in history)
        rmse_bh  = self._rmse(image_bh,       image_ideal)
        rmse_cor = self._rmse(image_corrected, image_ideal)
        improv   = (1.0 - rmse_cor / rmse_bh) * 100 if rmse_bh > 0 else 0.0

        fig.text(0.5, 0.965,
                 "Iterative Bone Beam-Hardening Correction",
                 ha="center", va="top", fontsize=15, fontweight="bold",
                 color=self._FG, fontfamily="monospace")
        fig.text(0.5, 0.945,
                 f"iterations = {n_iter}   |   "
                 f"image RMSE:  BH = {rmse_bh:.5f}  ->  "
                 f"corrected = {rmse_cor:.5f}   ({improv:+.1f} %)",
                 ha="center", va="top", fontsize=9, color=self._ACCENT)

        vmin_img  = min(image_bh.min(), image_corrected.min(), image_ideal.min())
        vmax_img  = max(image_bh.max(), image_corrected.max(), image_ideal.max())
        vmax_sino = max(sino_bh.max(), sino_corrected.max(), sino_ideal.max())
        diff_bh   = image_bh        - image_ideal
        diff_cor  = image_corrected - image_ideal
        vlim_diff = max(abs(diff_bh).max(), abs(diff_cor).max())

        # row 0 > sinograms
        for col, (sino, title) in enumerate([
            (sino_bh,        "Sinogram — BH (corrupted)"),
            (sino_corrected, "Sinogram — Bone-corrected"),
            (sino_ideal,     "Sinogram — Ideal"),
        ]):
            ax = fig.add_subplot(gs[0, col])
            self._panel(ax)
            im = ax.imshow(sino, cmap=self._SINO_CMAP, vmin=0, vmax=vmax_sino,
                           aspect="auto", interpolation="antialiased")
            ax.set_title(title, color=self._FG, fontsize=9, fontweight="bold", pad=4)
            if col < 2:
                ax.text(0.5, -0.04,
                        f"RMSE vs ideal = {self._rmse(sino, sino_ideal):.5f}",
                        transform=ax.transAxes, ha="center", va="top",
                        fontsize=7.5, color=self._ACCENT)
            self._cbar(fig, ax, im)

        # row 1 >reconstructions
        for col, (img, title) in enumerate([
            (image_bh,        "Reconstruction — BH"),
            (image_corrected, "Reconstruction — Bone-corrected"),
            (image_ideal,     "Reconstruction — Ideal"),
        ]):
            ax = fig.add_subplot(gs[1, col])
            self._panel(ax)
            im = ax.imshow(img, cmap=self._CT_CMAP,
                           vmin=vmin_img, vmax=vmax_img,
                           aspect="equal", interpolation="antialiased")
            ax.set_title(title, color=self._FG, fontsize=9, fontweight="bold", pad=4)
            if col < 2:
                ax.text(0.5, -0.04,
                        f"RMSE vs ideal = {self._rmse(img, image_ideal):.5f}",
                        transform=ax.transAxes, ha="center", va="top",
                        fontsize=7.5, color=self._ACCENT)
            self._cbar(fig, ax, im)

        # row2 >difference maps + bone mask
        norm_diff = Normalize(vmin=-vlim_diff, vmax=vlim_diff)
        for col, (diff, title, color) in enumerate([
            (diff_bh,  "Difference — BH minus Ideal",       self._RED),
            (diff_cor, "Difference — Corrected minus Ideal", self._GREEN),
        ]):
            ax = fig.add_subplot(gs[2, col])
            self._panel(ax)
            im = ax.imshow(diff, cmap=self._DIFF_CMAP, norm=norm_diff,
                           aspect="equal", interpolation="antialiased")
            ax.set_title(title, color=self._FG, fontsize=9, fontweight="bold", pad=4)
            ax.text(0.5, -0.04,
                    f"RMS error = {float(np.sqrt(np.mean(diff**2))):.5f}",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=7.5, color=color)
            self._cbar(fig, ax, im, label="delta intensity")

        ax_mask = fig.add_subplot(gs[2, 2])
        self._panel(ax_mask)
        ax_mask.imshow(image_ideal, cmap=self._CT_CMAP,
                       vmin=vmin_img, vmax=vmax_img,
                       aspect="equal", interpolation="antialiased")
        if bone_mask is not None:
            overlay = np.ma.masked_where(~bone_mask, np.ones_like(bone_mask, float))
            ax_mask.imshow(overlay, cmap=self._MASK_CMAP, alpha=0.55,
                           aspect="equal", vmin=0, vmax=1)
            n_px = int(bone_mask.sum())
            sub  = f"bone pixels: {n_px:,}  ({n_px/bone_mask.size*100:.1f}%)"
        else:
            sub = "no bone mask"
        ax_mask.set_title("Bone Mask on Ideal", color=self._FG,
                          fontsize=9, fontweight="bold", pad=4)
        ax_mask.text(0.5, -0.04, sub, transform=ax_mask.transAxes,
                     ha="center", va="top", fontsize=7.5, color=self._GREEN)

        # row 3 > convergence curves
        iters = [h["k"] for h in history]
        def _extract(key):
            return [h[key] if h[key] is not None else np.nan for h in history]

        specs = [
            (gs[3,0], _extract("delta_sino"),        True,  "Convergence — delta sinogram",
             "Iteration k", "RMSE(S^k,S^{k-1})/RMS(S_raw)", self._ACCENT, "o"),
            (gs[3,1], _extract("rmse_vs_ref_sino"),  False, "Sinogram RMSE vs Ideal",
             "Iteration k", "RMSE", self._ACCENT, "o"),
            (gs[3,2], _extract("rmse_vs_ref_image"), False, "Image RMSE vs Ideal",
             "Iteration k", "RMSE", self._GREEN,  "s"),
        ]
        for gs_cell, vals, do_log, title, xlabel, ylabel, color, marker in specs:
            ax = fig.add_subplot(gs_cell)
            ax.set_facecolor(self._PANEL)
            for sp in ax.spines.values(): sp.set_edgecolor(self._BORDER)
            valid = [(i,v) for i,v in zip(iters, vals) if not np.isnan(v)]
            if valid:
                xi, yi = zip(*valid)
                (ax.semilogy if do_log else ax.plot)(
                    xi, yi, color=color, marker=marker,
                    markersize=5, linewidth=1.8, markerfacecolor=self._BG)
            ax.set_title(title, color=self._FG, fontsize=9, fontweight="bold")
            ax.set_xlabel(xlabel, color=self._FG, fontsize=8)
            ax.set_ylabel(ylabel, color=self._FG, fontsize=7.5)
            ax.tick_params(colors=self._FG, labelsize=7)

        fig.savefig(self.output_path, dpi=self.dpi,
                    bbox_inches="tight", facecolor=self._BG)
        plt.close(fig)
        print(f"  Figure saved -> {self.output_path}")