from scipy.spatial import distance
import numpy as np

np.random.seed(1995)
# A = np.array([0.57955012, 0.9640571, 0.96120518]).reshape(-1, 1)
A = np.array([1, 2, 3]).reshape(-1, 1)
euc_dist = distance.pdist(A, metric="euclidean")
seu_dist = distance.pdist(A, metric="seuclidean")
print(f"Euclidean distance = {euc_dist} \nScaled Euclidean distance = {seu_dist}")
var = np.var(A, axis=0, ddof=1)
print(
    f"var = {var} and sqrt(var) = {np.sqrt(var)}"
    f"\nscaled Euclidean distance is: {euc_dist/np.sqrt(var)}"
)
