from pj_io import read_pj
import numpy as np
sino = read_pj("sino_BH.pj")
print("Shape:", sino.shape)
print("Min :", sino.min())
print("Max :", sino.max())
print("Mean:", sino.mean())
print("99th percentile:", np.percentile(sino, 99))
print("Number of NaNs/Inf:", np.isnan(sino).sum() + np.isinf(sino).sum())
