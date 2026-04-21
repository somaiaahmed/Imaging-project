import numpy as np


# ── Format constants ────────────────────────────────────────────────────────
# .pj files: optional 16-bit integer header followed by int16 projection data.
# read_pj auto-detects the header size; write_pj writes a clean file with
# NO header so files it creates are always readable at offset=0.
INT16_MAX = 32767


def read_pj(filename, nview=360, ndet=512, scale=None):
    """
    Read a raw .pj sinogram file.

    The file is assumed to store int16 samples, possibly preceded by a
    header of unknown size.  We try every offset in [0, 2000) and keep
    the one whose chunk has the highest standard deviation (most detail).

    Parameters
    ----------
    filename : str
        Path to the .pj file.
    nview : int
        Number of projection angles (rows).  Default 360.
    ndet : int
        Number of detector bins (columns).  Default 512.
    scale : float or None
        Divide the int16 values by this number to get physical units.
        If None the function auto-scales so the maximum absolute value
        in the returned array is 1.0.

    Returns
    -------
    sino : np.ndarray, shape (nview, ndet), dtype float32
        Sinogram values in [−1, 1] (auto-scale) or physical units.
    meta : dict
        {'offset': int, 'std_score': float, 'raw_scale': float}
    """
    raw = np.fromfile(filename, dtype=np.int16)
    expected = nview * ndet

    print(f"Total int16 values : {raw.size}")
    print(f"Expected           : {expected}  ({nview}×{ndet})")

    if raw.size < expected:
        raise ValueError(
            f"File too small: need {expected} int16 values, got {raw.size}."
        )

    best_score = -1.0
    best_offset = 0
    best_chunk = None

    for offset in range(0, min(2000, raw.size - expected + 1)):
        chunk = raw[offset: offset + expected]
        score = float(np.std(chunk))
        if score > best_score:
            best_score = score
            best_offset = offset
            best_chunk = chunk

    data = best_chunk.reshape((nview, ndet)).astype(np.float32)

    # Determine scale factor
    abs_max = np.max(np.abs(data))
    if abs_max == 0:
        raw_scale = 1.0
    elif scale is None:
        raw_scale = abs_max          # auto: map to [−1, 1]
    else:
        raw_scale = float(scale)

    data /= raw_scale

    print(f"Best offset        : {best_offset}")
    print(f"Std score (int16)  : {best_score:.4f}")
    print(f"Scale factor       : {raw_scale:.6f}")
    print(f"Shape              : {data.shape}")
    print(f"Min / Max          : {data.min():.6f} / {data.max():.6f}")

    meta = {"offset": best_offset, "std_score": best_score, "raw_scale": raw_scale}
    return data, meta


def write_pj(filename, data, raw_scale=None):
    """
    Write a sinogram to a .pj file.

    The data are stored as int16 with NO header, so the file is always
    readable by read_pj at offset 0.

    Parameters
    ----------
    filename : str
        Destination path.
    data : np.ndarray, shape (nview, ndet), dtype float32 (or any float)
        Sinogram values.  Values outside [−1, 1] are clipped before
        conversion.
    raw_scale : float or None
        Multiply `data` by this value before quantising to int16.
        If None the function uses INT16_MAX (32 767) so the full dynamic
        range is exploited.
    """
    if raw_scale is None:
        raw_scale = INT16_MAX

    quantised = np.clip(data * raw_scale, -INT16_MAX, INT16_MAX)
    quantised.astype(np.int16).tofile(filename)

    print(f"Written {filename!r}: shape={data.shape}, scale={raw_scale}")
