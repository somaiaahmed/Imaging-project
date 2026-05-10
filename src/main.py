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
from pj_io import INT16_MAX

NVIEW = 360
NDET = 512

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PATHS = {
    "raw": os.path.join(BASE_DIR, "data", "raw"),
    "gen": os.path.join(BASE_DIR, "data", "generated"),
    "figures": os.path.join(BASE_DIR, "data", "Figures"),
}

FILES = {
    "sino_ideal": os.path.join(PATHS["raw"], "sino_ideal.pj"),
    "sino_bh": os.path.join(PATHS["raw"], "sino_BH.pj"),

    "calibration": os.path.join(PATHS["gen"], "calibration.npz"),
    "sino_corrected": os.path.join(PATHS["gen"], "sino_corrected.pj"),
    "fig_stage1": os.path.join(PATHS["figures"], "correction_summary.png"),

    "sino_stage2": os.path.join(PATHS["gen"], "sino_stage2.pj"),
    "lut_npz": os.path.join(PATHS["gen"], "lut_stage2.npz"),
    "metrics_npz": os.path.join(PATHS["gen"], "stage2_metrics.npz"),
    "fig_stage2": os.path.join(PATHS["figures"], "stage2_summary.png"),
    
    "sino_bone":    os.path.join(PATHS["gen"], "sino_bone_corrected.pj"),
    "bone_metrics": os.path.join(PATHS["gen"], "bone_metrics.npz"),
    "fig_bone":     os.path.join(PATHS["figures"], "bone_correction_summary.png"),
}


def ensure_dirs():
    for p in PATHS.values():
        os.makedirs(p, exist_ok=True)


def run_generate():
    """
    Note --> both files must use the same scale factor so they can be comparable for calibration
    scale is fixed to INT16_MAX / ideal_max so ideal maps to [0,1]
    BH is proportionally smaller (reflecting the real suppression)
    """
    ensure_dirs()

    phantom = SheppLoganPhantom(size=NDET)
    projector = ForwardProjector(n_views=NVIEW, n_det=NDET)
    bh_sim = BeamHardeningSimulator(order=3, severity="strong")
    sino_ideal = projector.project(phantom.image)
    sino_bh = bh_sim.apply(sino_ideal)

    shared_scale = INT16_MAX / float(sino_ideal.max())
    pj_io.write_pj(FILES["sino_ideal"], sino_ideal, raw_scale=shared_scale)
    pj_io.write_pj(FILES["sino_bh"],    sino_bh,    raw_scale=shared_scale)

    print("Successfully generated sinograms")


"""
Stage #1
teach the program how to undo beam hardening using a curve fit
calibration --> learn the coefficients and save to use them later on new corrupted sinograms
Correction --> apply that polynomial to every pixel in the corrupted sinogram

"""
def run_calibrate():
    sino_bh, _ = pj_io.read_pj(FILES["sino_bh"], NVIEW, NDET)
    sino_ideal, _ = pj_io.read_pj(FILES["sino_ideal"], NVIEW, NDET)

    calibrator = BHCalibrator(degree=3, subsample=4)
    calibrator.fit(sino_bh, sino_ideal)
    calibrator.save(FILES["calibration"])
    print("calibration saved")


def run_correct(show_plot=True):
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

    print("correction done")


"""
Stage #2
A LUT is just a table that says 
if input is this value -> output should be that value
so Instead of using one polynomial formula for all values, LUT uses a stored table and interpolation between points. 

Steps
1. take a BH value 
2. look up the corrected value from a table 
3. use interpolation if the exact value is not listed 
"""
def run_build_lut(input_file=None):
    if input_file is None:
        input_file = FILES["sino_bh"]

    sino_bh, _ = pj_io.read_pj(input_file, NVIEW, NDET)
    sino_ideal, _ = pj_io.read_pj(FILES["sino_ideal"], NVIEW, NDET)

    spectrum = XRaySpectrum(kVp=80)

    physics_lut = PhysicsLUT(spectrum)
    empirical_lut = EmpiricalLUT(sino_bh, sino_ideal)
    blended_lut = BlendedLUT(empirical_lut, physics_lut, alpha=0.9)

    np.savez(FILES["lut_npz"],
             physics_lut=physics_lut.table,
             empirical_lut=empirical_lut.table,
             blended_lut=blended_lut.table)

    print("LUTs built and saved")


def run_apply_lut(input_file=None):
    if input_file is None:
        input_file = FILES["sino_bh"]

    sino_bh, _ = pj_io.read_pj(input_file, NVIEW, NDET)
    sino_ideal, _ = pj_io.read_pj(FILES["sino_ideal"], NVIEW, NDET)

    lut_data = np.load(FILES["lut_npz"], allow_pickle=True)

    empirical_lut = EmpiricalLUT.from_table(lut_data["empirical_lut"])
    blended_lut = BlendedLUT.from_table(lut_data["blended_lut"])

    sino_emp = empirical_lut.apply(sino_bh)
    sino_comb = blended_lut.apply(sino_bh)

    pj_io.write_pj(FILES["sino_stage2"], sino_emp)

    np.savez(FILES["metrics_npz"],
             sino_emp=sino_emp,
             sino_comb=sino_comb)

    print("LUT correction applied")


def run_reconstruct():
    sino_ideal, _ = pj_io.read_pj(FILES["sino_ideal"], NVIEW, NDET)
    sino_bh, _ = pj_io.read_pj(FILES["sino_bh"], NVIEW, NDET)
    sino_stage1, _ = pj_io.read_pj(FILES["sino_corrected"], NVIEW, NDET)

    metrics = np.load(FILES["metrics_npz"])
    sino_emp = metrics["sino_emp"]
    sino_comb = metrics["sino_comb"]

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

    print("reconstruction and evaluation done")


def run_stage2_plot(show_plot=True):
    if not show_plot:
        return

    sino_ideal,_ = pj_io.read_pj(FILES["sino_ideal"],     NVIEW, NDET)
    sino_bh, _ = pj_io.read_pj(FILES["sino_bh"],        NVIEW, NDET)
    sino_stage1, _ = pj_io.read_pj(FILES["sino_corrected"], NVIEW, NDET)
    sino_stage2, _ = pj_io.read_pj(FILES["sino_stage2"],    NVIEW, NDET)

    lut_data = np.load(FILES["lut_npz"], allow_pickle=True)
    metrics = np.load(FILES["metrics_npz"])
    sino_comb = metrics["sino_comb"]

    physics_lut = PhysicsLUT.from_table(lut_data["physics_lut"])
    empirical_lut = EmpiricalLUT.from_table(lut_data["empirical_lut"])

    recon = FBPReconstructor()
    rec = recon.reconstruct_many(
        ideal    = sino_ideal,
        bh       = sino_bh,
        stage1   = sino_stage1,
        empirical= sino_stage2,
        combined = sino_comb,
    )

    def _rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))

    sinograms = {
        "bh":      (sino_bh,     "BH (corrupted)",    0.0),
        "stage1":  (sino_stage1, "Stage 1 corrected", 0.0),
        "empirical":(sino_stage2,"Stage 2 LUT",       0.0),
        "ideal":   (sino_ideal,  "Ideal",             0.0),
    }
    images = {
        "bh":      (rec["bh"],        "BH",      _rmse(rec["bh"],        rec["ideal"])),
        "stage1":  (rec["stage1"],    "Stage 1", _rmse(rec["stage1"],    rec["ideal"])),
        "empirical":(rec["empirical"],"Stage 2", _rmse(rec["empirical"], rec["ideal"])),
        "ideal":   (rec["ideal"],     "Ideal",   0.0),
    }

    plotter = Stage2Plotter(
        spectrum      = XRaySpectrum(kVp=80),
        physics_lut   = physics_lut,
        empirical_lut = empirical_lut,
        output_path   = FILES["fig_stage2"],
    )
    plotter.plot(
        sinograms         = sinograms,
        images            = images,
        sino_bh           = sino_bh,
        sino_stage1       = sino_stage1,
        sino_lut_combined = sino_comb,
        sino_ideal        = sino_ideal,
        recon_ideal       = rec["ideal"],
    )

    print("stage 2 plot saved")


def run_stage2(input_file=None):
    run_build_lut(input_file)
    run_apply_lut(input_file)
    run_reconstruct()
    run_stage2_plot()

# ===========  stage3 the bone correction method ===========
def run_bone_correct(show_plot: bool = True):
    """
    Stage 3: iterative bone beam-hardening correction 
    """
    print("\n=== Stage 3: Bone Correction ===")
    ensure_dirs()
 
    sino_bh,    _ = pj_io.read_pj(FILES["sino_bh"],    NVIEW, NDET)
    sino_ideal, _ = pj_io.read_pj(FILES["sino_ideal"], NVIEW, NDET)
 
    # reconstruct the ideal image for evaluation only
    from reconstruction import FBPReconstructor
    ref_image = FBPReconstructor().reconstruct(sino_ideal)
    image_bh  = FBPReconstructor().reconstruct(sino_bh)
 
    from bone_correction import IterativeBoneCorrector, BoneCorrectionPlotter
 
    corrector = IterativeBoneCorrector(
        n_views      = NVIEW,
        n_det        = NDET,
        severity     = "strong",   # BeamHardeningSimulator severity
        max_iter     = 14,
        tol          = 1e-5,
        lam          = 0.20,
        hu_threshold = 300.0,
        erode_px     = 2,
        verbose      = True,
    )
 
    result = corrector.correct(
        sino_raw        = sino_bh,
        reference_sino  = sino_ideal,
        reference_image = ref_image,
    )
 
    corrector.print_summary()
 
    # saving
    pj_io.write_pj(FILES["sino_bone"], result["sino_corrected"])
    history = result["history"]
    np.savez(
        FILES["bone_metrics"],
        sino_corrected  = result["sino_corrected"],
        image_corrected = result["image_corrected"],
        image_initial   = result["image_initial"],
        converged       = result["converged"],
        n_iter          = result["n_iter"],
        delta_sino      = np.array([
            h["delta_sino"] if h["delta_sino"] is not None else np.nan
            for h in history
        ]),
        rmse_sino       = np.array([
            h["rmse_vs_ref_sino"] if h["rmse_vs_ref_sino"] is not None else np.nan
            for h in history
        ]),
        rmse_image      = np.array([
            h["rmse_vs_ref_image"] if h["rmse_vs_ref_image"] is not None else np.nan
            for h in history
        ]),
    )
 
    print(f"\n  Converged : {result['converged']}")
    print(f"  Iterations: {result['n_iter']}")
 
    if show_plot:
        bone_mask = None
        for h in reversed(history):
            if h["bone_mask"] is not None:
                bone_mask = h["bone_mask"]
                break
 
        plotter = BoneCorrectionPlotter(
            output_path = FILES["fig_bone"],
            dpi         = 150,
        )
        plotter.plot(
            sino_bh         = sino_bh,
            sino_corrected  = result["sino_corrected"],
            sino_ideal      = sino_ideal,
            image_bh        = image_bh,
            image_corrected = result["image_corrected"],
            image_ideal     = ref_image,
            bone_mask       = bone_mask,
            history         = history,
        )
 
    print("bone correction done")

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
    run_bone_correct(False)
    run_stage2(input_file=FILES["sino_corrected"])
    print("\n ALL DONE")
    print("\n ALL DONE")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=[
        "generate", "calibrate", "correct",
        "build_lut", "apply_lut", "reconstruct", "stage2_plot",
        "stage2", "show", "all", "bone_correct"
    ])
    parser.add_argument("files", nargs="*")

    args = parser.parse_args()

    if args.stage == "generate":
        run_generate()
    elif args.stage == "calibrate":
        run_calibrate()
    elif args.stage == "correct":
        run_correct()
    elif args.stage == "build_lut":
        run_build_lut()
    elif args.stage == "apply_lut":
        run_apply_lut()
    elif args.stage == "reconstruct":
        run_reconstruct()
    elif args.stage == "stage2_plot":
        run_stage2_plot()
    elif args.stage == "stage2":
        run_stage2()
    elif args.stage == "show":
        run_show(args.files if args.files else [FILES["sino_bh"]])
    elif args.stage == "all":
        run_all()
    elif args.stage == "bone_correct":
        run_bone_correct(True)           


if __name__ == "__main__":
    main()

