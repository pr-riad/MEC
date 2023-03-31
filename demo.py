import numpy as np
from sklearn.datasets import make_blobs
from mec import mec_eval


n_samples = 1500
random_state = 170
X, _ = make_blobs(n_samples=n_samples, random_state=random_state)

