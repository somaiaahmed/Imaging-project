from pj_io import read_pj

sino = read_pj("sino_BH.pj")

print("Sinogram loaded successfully!")
print("Shape:", sino.shape)
print("Min:", sino.min())
print("Max:", sino.max())
