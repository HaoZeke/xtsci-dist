import numpy as np
from scipy.spatial import distance

np.random.seed(1995)
# A = np.array([0.57955012, 0.9640571, 0.96120518]).reshape(-1, 1)
# A = np.array([1, 2, 3]).reshape(-1, 1)
inp = np.random.rand(3, 4)
print(inp)
eucDist = distance.pdist(inp, metric="euclidean")
seuDist = distance.pdist(inp, metric="seuclidean")
print(f"Euclidean distance = {eucDist} \nScaled Euclidean distance = {seuDist}")
var = np.var(inp, axis=0, ddof=1)
print(
    f"var = {var} and sqrt(var) = {np.sqrt(var)}"
    # f"\nscaled Euclidean distance is: {eucDist/np.sqrt(var)}"
)
np.savez_compressed("inp_3_4.npz", inp=inp)
np.savez_compressed("eucDist_3_4.npz", eucDist=eucDist)
np.savez_compressed("seuDist_3_4.npz", seuDist=seuDist)
