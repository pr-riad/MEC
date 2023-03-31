import numpy as np
from sklearn.datasets import make_blobs
from mec import mec_eval


n_samples = 1000
random_state = 41
n_centers = 8
X, y = make_blobs(n_samples=n_samples, centers = n_centers, random_state=random_state)

k = mec_eval(X, k_min = 2, k_max = 10, disp = True) 

print()
print("predecte n_cluster = ", k)
print("GT n_cluster = ", len(np.unique(y)))
