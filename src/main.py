"""
main.py — Clean version
CT Beam Hardening Pipeline
"""

from __future__ import annotations
import argparse
import os
import numpy as np

import pj_io
from phantom import SheppLoganPhantom, ForwardProjector, BeamHardeningSimulator
from calibration import BHCalibrator, BHCorrector
from spectrum import XRaySpectrum
from lut import PhysicsLUT, EmpiricalLUT, BlendedLUT
from reconstruction import FBPReconstructor, Evaluator
from plotter import Stage1Plotter, Stage2Plotter, SinogramViewer


# ════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════

NVIEW = 360
NDET = 512

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PATHS = {
    "raw": os.path.join(BASE_DIR, "data", "raw"),
    "gen": os.path.join(BASE_DIR, "data", "generated"),
}

FILES = {
    "sino_ideal": os.path.join(PATHS["raw"], "sino_ideal.pj"),
    "sino_bh": os.path.join(PATHS["raw"], "sino_BH.pj"),

    "calibration": os.path.join(PATHS["gen"], "calibration.npz"),
    "sino_corrected": os.path.join(PATHS["gen"], "sino_corrected.pj"),
    "fig_stage1": os.path.join(PATHS["gen"], "correction_summary.png"),

    "sino_stage2": os.path.join(PATHS["gen"], "sino_stage2.pj"),
    "lut_npz": os.path.join(PATHS["gen"], "lut_stage2.npz"),
    "metrics_npz": os.path.join(PATHS["gen"], "stage2_metrics.npz"),
    "fig_stage2": os.path.join(PATHS["gen"], "stage2_summary.png"),
}


def ensure_dirs():
    for p in PATHS.values():
        os.makedirs(p, exist_ok=True)


# ════════════════════════════════════════════════════════════
# STAGE 1
# ════════════════════════════════════════════════════════════

def run_generate():
    print("\n=== GENERATE ===")
    ensure_dirs()

    phantom = SheppLoganPhantom(size=NDET)
    projector = ForwardProjector(n_views=NVIEW, n_det=NDET)
    bh_sim = BeamHardeningSimulator(order=3)

    sino_ideal = projector.project(phantom.image)
    sino_bh = bh_sim.apply(sino_ideal)

    pj_io.write_pj(FILES["sino_ideal"], sino_ideal)
    pj_io.write_pj(FILES["sino_bh"], sino_bh)

    print("✔ Generated sinograms")


def run_calibrate():
    print("\n=== CALIBRATE ===")

    sino_bh, _ = pj_io.read_pj(FILES["sino_bh"], NVIEW, NDET)
    sino_ideal, _ = pj_io.read_pj(FILES["sino_ideal"], NVIEW, NDET)

    calibrator = BHCalibrator(degree=3, subsample=4)
    calibrator.fit(sino_bh, sino_ideal)
    calibrator.save(FILES["calibration"])

    print("✔ Calibration saved")


def run_correct(show_plot=True):
    print("\n=== CORRECT ===")

    corrector = BHCorrector(FILES["calibration"])

    sino_bh, _ = pj_io.read_pj(FILES["sino_bh"], NVIEW, NDET)
    sino_ideal, _ = pj_io.read_pj(FILES["sino_ideal"], NVIEW, NDET)

    sino_corrected = corrector.correct(sino_bh)
    corrector.evaluate(sino_bh, sino_corrected, sino_ideal)

    pj_io.write_pj(FILES["sino_corrected"], sino_corrected)

    if show_plot:
        cal = np.load(FILES["calibration"])
        plotter = Stage1Plotter(FILES["fig_stage1"])
        plotter.plot(
            sinos={
                "ideal": sino_ideal,
                "bh": sino_bh,
                "corrected": sino_corrected,
            },
            calibration={
                "coeffs": cal["coeffs"],
                "degree": int(cal["degree"]),
                "r2": float(cal["r2"]),
            },
            n_views=NVIEW,
        )

    print("✔ Correction done")


# ════════════════════════════════════════════════════════════
# STAGE 2
# ════════════════════════════════════════════════════════════

def run_stage2():
    print("\n=== STAGE 2 ===")

    sino_stage1, _ = pj_io.read_pj(FILES["sino_corrected"], NVIEW, NDET)
    sino_ideal, _ = pj_io.read_pj(FILES["sino_ideal"], NVIEW, NDET)
    sino_bh, _ = pj_io.read_pj(FILES["sino_bh"], NVIEW, NDET)

    spectrum = XRaySpectrum(kVp=80)

    physics_lut = PhysicsLUT(spectrum)
    empirical_lut = EmpiricalLUT(sino_bh, sino_ideal)

    blended_lut = BlendedLUT(empirical_lut, physics_lut, alpha=0.9)

    sino_emp = empirical_lut.apply(sino_bh)
    sino_comb = blended_lut.apply(sino_bh)

    recon = FBPReconstructor()
    rec = recon.reconstruct_many(
        ideal=sino_ideal,
        bh=sino_bh,
        stage1=sino_stage1,
        empirical=sino_emp,
        combined=sino_comb,
    )

    evaluator = Evaluator(sino_ideal, rec["ideal"])

    evaluator.report(
        evaluator.evaluate_sinograms(
            bh=sino_bh, stage1=sino_stage1, empirical=sino_emp, combined=sino_comb
        ),
        evaluator.evaluate_images(
            bh=rec["bh"], stage1=rec["stage1"], final=rec["empirical"]
        ),
    )

    pj_io.write_pj(FILES["sino_stage2"], sino_stage1)

    print("✔ Stage 2 complete")


# ════════════════════════════════════════════════════════════
# UTILS
# ════════════════════════════════════════════════════════════

def run_show(files):
    sinos = {}
    for f in files:
        try:
            sino, _ = pj_io.read_pj(f, NVIEW, NDET)
            sinos[f] = sino
        except:
            print(f"Missing: {f}")

    if sinos:
        SinogramViewer().show(sinos)


def run_all():
    run_generate()
    run_calibrate()
    run_correct(False)
    run_stage2()
    print("\n✔ ALL DONE")


# ════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=[
        "generate", "calibrate", "correct",
        "stage2", "show", "all"
    ])
    parser.add_argument("files", nargs="*")

    args = parser.parse_args()

    if args.stage == "generate":
        run_generate()
    elif args.stage == "calibrate":
        run_calibrate()
    elif args.stage == "correct":
        run_correct()
    elif args.stage == "stage2":
        run_stage2()
    elif args.stage == "show":
        run_show(args.files if args.files else [FILES["sino_bh"]])
    elif args.stage == "all":
        run_all()


if __name__ == "__main__":
    main()