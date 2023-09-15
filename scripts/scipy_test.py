import numpy as np
from scipy.spatial import distance

np.random.seed(1995)
# pdist
# # A = np.array([0.57955012, 0.9640571, 0.96120518]).reshape(-1, 1)
# # A = np.array([1, 2, 3]).reshape(-1, 1)
# inp = np.random.rand(3, 4)
# print(inp)
# eucDist = distance.pdist(inp, metric="euclidean")
# seuDist = distance.pdist(inp, metric="seuclidean")
# sqeuDist = distance.pdist(inp, metric="sqeuclidean")
# print(
#     f"Euclidean distance = {eucDist} \n"
#     f"Scaled Euclidean distance = {seuDist}\n"
#     f"Sq. Euclidean distance = {sqeuDist}"
# )
# print(f"Euc Sq. = {eucDist**2}")
# var = np.var(inp, axis=0, ddof=1)
# print(
#     f"var = {var} and sqrt(var) = {np.sqrt(var)}"
#     # f"\nscaled Euclidean distance is: {eucDist/np.sqrt(var)}"
# )
# np.savez_compressed("inp_3_4.npz", inp=inp)
# np.savez_compressed("eucDist_3_4.npz", eucDist=eucDist)
# np.savez_compressed("seuDist_3_4.npz", seuDist=seuDist)
# np.savez_compressed("sqeuDist_3_4.npz", sqeuDist=sqeuDist)

# cdist
inpa = np.random.rand(3, 4)
inpb = np.random.rand(2, 4)
print(f"inpa: {inpa}")
print(f"inpb: {inpb}")
eucCdist = distance.cdist(inpa, inpb, metric="euclidean")
print(f"Euclidean cdist: {eucCdist}")

np.savez_compressed("inpa_3_4.npz", inpa=inpa)
np.savez_compressed("inpb_2_4.npz", inpb=inpb)
np.savez_compressed("eucCdist_3_4_2_4.npz", eucCdist=eucCdist)
