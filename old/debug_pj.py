import numpy as np

filename = "sino_BH.pj"

for dtype in [np.int16, np.uint16, np.float32, np.float64]:
    raw = np.fromfile(filename, dtype=dtype)
    print("\n---", dtype, "---")
    print("first 10:", raw[:10])
    print("min/max:", raw.min(), raw.max())
