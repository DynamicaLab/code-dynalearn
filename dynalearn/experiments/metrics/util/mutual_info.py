import numpy as np

from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors, KDTree


def mutual_info(x, y, n_neighbors=3, metric="euclidean"):
    n_samples = x.shape[0]
    if n_samples != y.shape[0]:
        raise ValueError(
            f"Invalid shapes: got {x.shape} for `x` and {y.shape} for `y`."
        )
    xy = np.hstack((x, y))

    # Here we rely on NearestNeighbors to select the fastest algorithm.
    nn = NearestNeighbors(metric=metric, n_neighbors=n_neighbors)

    nn.fit(xy)
    radius = nn.kneighbors()[0]
    radius = np.nextafter(radius[:, -1], 0)

    # KDTree is explicitly fit to allow for the querying of number of
    # neighbors within a specified radius
    kd = KDTree(x, metric=metric)
    nx = kd.query_radius(x, radius, count_only=True, return_distance=False)
    nx = np.array(nx) - 1.0

    kd = KDTree(y, metric=metric)
    ny = kd.query_radius(y, radius, count_only=True, return_distance=False)
    ny = np.array(ny) - 1.0

    mi = (
        digamma(n_samples)
        + digamma(n_neighbors)
        - np.mean(digamma(nx + 1))
        - np.mean(digamma(ny + 1))
    )
    return max(0, mi)
