"""
main — Orchestrator for Stage 2: LUT Correction + Image Enhancement.

Run
---
    python main.py

Inputs  (current directory)
-------
    sino_corrected.pj   Stage 1 corrected sinogram
    sino_ideal.pj       Ground-truth monoenergetic sinogram
    sino_BH.pj          Raw beam-hardened sinogram

Outputs (current directory)
--------
    sino_stage2.pj      Sinogram forwarded to Stage 3
    lut_stage2.npz      LUT data arrays
    stage2_metrics.npz  RMSE metrics
    stage2_summary.png  Summary figure
"""

import numpy as np
import pj_io

from spectrum        import XRaySpectrum
from lut             import PhysicsLUT, EmpiricalLUT, BlendedLUT
from reconstruction  import FBPReconstructor, CuppingCorrector, Evaluator
from plotter         import Stage2Plotter


def load_sinograms() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and unpack the three input sinograms."""
    print("\nLoading sinograms...")

    def _unpack(data):
        return data[0] if isinstance(data, tuple) else data

    sino_stage1 = _unpack(pj_io.read_pj("./data/generated/sino_corrected.pj"))
    sino_ideal  = _unpack(pj_io.read_pj("./data/raw/sino_ideal.pj"))
    sino_bh     = _unpack(pj_io.read_pj("./data/raw/sino_BH.pj"))

    print(f"    Shape      : {sino_stage1.shape}  (views x detectors)")
    print(f"    sino_BH    : {sino_bh.min():.4f} -> {sino_bh.max():.4f}")
    print(f"    sino_stage1: {sino_stage1.min():.4f} -> {sino_stage1.max():.4f}")
    print(f"    sino_ideal : {sino_ideal.min():.4f} -> {sino_ideal.max():.4f}")

    return sino_stage1, sino_ideal, sino_bh


def save_outputs(
    sino_stage1: np.ndarray,
    empirical_lut: EmpiricalLUT,
    physics_lut: PhysicsLUT,
    spectrum: XRaySpectrum,
    sino_metrics: dict,
    image_metrics: dict,
) -> None:
    """Persist sinogram, LUT arrays, and metrics to disk."""

    pj_io.write_pj("sino_stage2.pj", sino_stage1)
    print("    Saved -> sino_stage2.pj  (for Stage 3)")

    np.savez(
        "lut_stage2.npz",
        bh_centers   = empirical_lut.bh_centers,
        ideal_means  = empirical_lut.ideal_means,
        p_poly_norm  = physics_lut.p_poly_norm,
        p_ideal_norm = physics_lut.p_ideal_norm,
        E            = spectrum.E,
        S            = spectrum.S,
        mu_water     = spectrum.mu_water,
    )
    print("    Saved -> lut_stage2.npz")

    np.savez(
        "stage2_metrics.npz",
        **{f"sino_{k}": v  for k, v in sino_metrics.items()},
        **{f"img_{k}":  v  for k, v in image_metrics.items()},
        improvement_pct = (
            (1 - image_metrics["final"] / image_metrics["bh"]) * 100
            if "bh" in image_metrics and "final" in image_metrics
            else np.nan
        ),
    )
    print("    Saved -> stage2_metrics.npz")


def main() -> None:

    # ── 1. Load data ───────────────────────────────────────────────────────────
    sino_stage1, sino_ideal, sino_bh = load_sinograms()

    # ── 2. Build spectrum model ────────────────────────────────────────────────
    print("\nBuilding X-ray spectrum...")
    spectrum = XRaySpectrum(kVp=80)
    spectrum.summary()

    # ── 3. Build LUTs ─────────────────────────────────────────────────────────
    print("\nBuilding LUTs...")
    physics_lut   = PhysicsLUT(spectrum)
    empirical_lut = EmpiricalLUT(sino_bh, sino_ideal)

    # Blended: 90% empirical + 10% pass-through (regularises extreme values)
    class _PassThrough:
        def apply(self, sino): return sino
    blended_lut = BlendedLUT(empirical_lut, _PassThrough(), alpha=0.9)

    # ── 4. Apply LUT corrections ───────────────────────────────────────────────
    print("\nApplying LUT corrections...")
    sino_lut_empirical = empirical_lut.apply(sino_bh)
    sino_lut_combined  = blended_lut.apply(sino_bh)

    # ── 5. Reconstruct ─────────────────────────────────────────────────────────
    print("\nReconstructing images (FBP)...")
    recon = FBPReconstructor()
    reconstructions = recon.reconstruct_many(
        ideal     = sino_ideal,
        bh        = sino_bh,
        stage1    = sino_stage1,
        combined  = sino_lut_combined,
        empirical = sino_lut_empirical,
    )

    # ── 6. Cupping correction ──────────────────────────────────────────────────
    cupper = CuppingCorrector(strength=0.3)
    recon_combined_cup = cupper.correct(reconstructions["combined"])

    # Stage 2 final = empirical LUT reconstruction
    recon_final = reconstructions["empirical"]

    # ── 7. Evaluate ────────────────────────────────────────────────────────────
    print("\nEvaluating...")
    evaluator = Evaluator(
        reference_sino  = sino_ideal,
        reference_image = reconstructions["ideal"],
    )

    sino_metrics = evaluator.evaluate_sinograms(
        bh        = sino_bh,
        stage1    = sino_stage1,
        empirical = sino_lut_empirical,
        combined  = sino_lut_combined,
    )
    image_metrics = evaluator.evaluate_images(
        bh     = reconstructions["bh"],
        stage1 = reconstructions["stage1"],
        final  = recon_final,
    )
    evaluator.report(sino_metrics, image_metrics,
                     baseline_key="bh", stage1_key="stage1", final_key="final")

    # ── 8. Save outputs ────────────────────────────────────────────────────────
    print("\nSaving outputs...")
    save_outputs(sino_stage1, empirical_lut, physics_lut, spectrum,
                 sino_metrics, image_metrics)

    # ── 9. Plot ────────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plotter = Stage2Plotter(spectrum, physics_lut, empirical_lut)

    sino_panel = {
        "bh":       (sino_bh,           f"Sino: Original BH\nRMSE={sino_metrics['bh']:.6f}",       sino_metrics["bh"]),
        "stage1":   (sino_stage1,       f"Sino: After Stage 1\nRMSE={sino_metrics['stage1']:.6f}", sino_metrics["stage1"]),
        "combined": (sino_lut_combined, f"Sino: Combined LUT\nRMSE={sino_metrics['combined']:.6f}",sino_metrics["combined"]),
        "ideal":    (sino_ideal,        "Sino: Ground Truth",                                        0.0),
    }
    img_panel = {
        "bh":     (reconstructions["bh"],    f"Image: Original BH\nRMSE={image_metrics['bh']:.6f}",    image_metrics["bh"]),
        "stage1": (reconstructions["stage1"],f"Image: Stage 1\nRMSE={image_metrics['stage1']:.6f}",    image_metrics["stage1"]),
        "final":  (recon_final,              f"Image: Stage 2 Final\nRMSE={image_metrics['final']:.6f}",image_metrics["final"]),
        "ideal":  (reconstructions["ideal"], "Image: Ground Truth",                                      0.0),
    }

    plotter.plot(
        sinograms         = sino_panel,
        images            = img_panel,
        sino_bh           = sino_bh,
        sino_stage1       = sino_stage1,
        sino_lut_combined = sino_lut_combined,
        sino_ideal        = sino_ideal,
        recon_ideal       = reconstructions["ideal"],
    )

    print("\nStage 2 complete.")


if __name__ == "__main__":
    main()
