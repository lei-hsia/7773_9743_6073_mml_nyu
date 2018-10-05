import numpy as np

from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import pairwise_distances

# 10000 examples, each represents a 2-dimensional variable
a = np.random.rand(3, 2)

b = np.random.rand(3, 2)

x = np.random.rand(1,2)

y = np.random.rand(1,2)

dist=np.linalg.norm(x-y)

