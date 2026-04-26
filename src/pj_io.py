import numpy as np


# ── Format constants ────────────────────────────────────────────────────────
# .pj files: optional 16-bit integer header followed by int16 projection data.
# read_pj auto-detects the header size; write_pj writes a clean file with
# NO header so files it creates are always readable at offset=0.
INT16_MAX = 32767


def read_pj(filename, nview=360, ndet=512, scale=None):
    """
    Read a raw .pj sinogram file written by write_pj.

    IMPORTANT: write_pj now always uses a shared scale (INT16_MAX /
    ideal_max) for both ideal and BH files.  read_pj therefore defaults
    to dividing by INT16_MAX so both files decode to the same float
    range and calibration can compare them directly.

    Parameters
    ----------
    filename : str
    nview, ndet : int
    scale : float or None
        Override divisor. Default: INT16_MAX (matches write_pj default).
    """
    raw = np.fromfile(filename, dtype=np.int16)
    expected = nview * ndet

    print(f"Total int16 values : {raw.size}")
    print(f"Expected           : {expected}  ({nview}×{ndet})")

    if raw.size < expected:
        raise ValueError(
            f"File too small: need {expected} int16 values, got {raw.size}."
        )

    # Try offsets 0..min(2000) and pick the one with highest std
    best_score, best_offset, best_chunk = -1.0, 0, None
    for offset in range(0, min(2000, raw.size - expected + 1)):
        chunk = raw[offset: offset + expected]
        score = float(np.std(chunk))
        if score > best_score:
            best_score, best_offset, best_chunk = score, offset, chunk

    data = best_chunk.reshape((nview, ndet)).astype(np.float32)

    # Use INT16_MAX as default so files written with the shared scale
    # decode correctly and BH values remain proportionally smaller
    raw_scale = float(scale) if scale is not None else float(INT16_MAX)
    data /= raw_scale

    print(f"Best offset        : {best_offset}")
    print(f"Scale factor       : {raw_scale:.2f}")
    print(f"Shape              : {data.shape}")
    print(f"Min / Max          : {data.min():.6f} / {data.max():.6f}")

    meta = {"offset": best_offset, "std_score": best_score, "raw_scale": raw_scale}
    return data, meta


def write_pj(filename, data, raw_scale=None):
    """
    Write a sinogram to a .pj file.

    Stored as int16 with NO header (always readable at offset 0).
    Uses a fixed scale of INT16_MAX so both ideal and BH sinograms
    are written with the same quantisation — critical for calibration.

    Parameters
    ----------
    filename : str
    data : np.ndarray, shape (nview, ndet), values in [0, 1]
    raw_scale : float or None
        Override scale factor. Default: INT16_MAX (32767).
    """
    if raw_scale is None:
        raw_scale = INT16_MAX

    quantised = np.clip(data * raw_scale, -INT16_MAX, INT16_MAX)
    quantised.astype(np.int16).tofile(filename)

    print(f"Written {filename!r}: shape={data.shape}, "
          f"range=[{data.min():.4f}, {data.max():.4f}], scale={raw_scale}")