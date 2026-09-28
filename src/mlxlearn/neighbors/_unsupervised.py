"""Unsupervised nearest neighbors."""

from __future__ import annotations

from sklearn.neighbors import NearestNeighbors as _SkNearestNeighbors

from .._common.arrays import NUMPY_FLOAT
from .._common.validation import check_is_fitted, is_sparse, validate_data
from ._base import MLXNeighborsMixin, _reject_string_data, _safe_shape

__all__ = ["NearestNeighbors"]


class NearestNeighbors(MLXNeighborsMixin, _SkNearestNeighbors):
    """Unsupervised nearest-neighbor search, accelerated with MLX.

    A drop-in replacement for :class:`sklearn.neighbors.NearestNeighbors` for
    exact brute-force Euclidean search. Parameters, fitted attributes and error
    messages follow scikit-learn.

    Parameters
    ----------
    n_neighbors : int, default=5
    radius : float, default=1.0
        Used by :meth:`radius_neighbors`, which runs on scikit-learn — see Notes.
    algorithm : {"auto", "brute"}, default="auto"
        Only brute-force search is accelerated. ``"kd_tree"`` and ``"ball_tree"``
        raise :class:`~mlxlearn.exceptions.UnsupportedParameterError` on direct
        use and fall back to scikit-learn under ``patch_sklearn()``. See Notes for
        how ``"auto"`` is dispatched on low-dimensional data.
    leaf_size : int, default=30
        Accepted for signature compatibility; brute-force search does not use it.
    metric : {"euclidean", "l2", "minkowski", "sqeuclidean"}, default="minkowski"
        ``"minkowski"`` is supported at ``p=2`` only.
    p : float, default=2
    metric_params : dict, default=None
        Only Minkowski's ``p``, which overrides the ``p`` parameter as it does in
        scikit-learn. Anything else — a weight vector ``w``, another metric's
        parameters — raises rather than being silently ignored, so mlxlearn never
        computes unweighted Euclidean distances for a weighted request.
    n_jobs : int, default=None
        Accepted and ignored; the MLX path does not use joblib parallelism.

    Attributes
    ----------
    n_samples_fit_ : int
    n_features_in_ : int
    effective_metric_ : str
        As scikit-learn sets it: ``"euclidean"`` for ``metric="minkowski"`` at
        ``p=2``, otherwise the metric as given.
    execution_backend_ : {"mlx", "cpu", "sklearn"}
        Which backend fitted this estimator. Every subsequent query uses it.

    Notes
    -----
    :meth:`radius_neighbors` and :meth:`radius_neighbors_graph` are **not**
    accelerated. mlxlearn keeps the fitted state scikit-learn's brute-force
    implementation reads, so they run scikit-learn's own code on the CPU — at
    stock speed, because the query is handed over in float32 to match the float32
    training copy. Distances are computed against that copy, so they carry the
    float32 rounding of the training data that ``docs/fp32_policy.md`` §3.4
    describes, exactly as :meth:`kneighbors` does. Stated rather than left for a
    user to discover from a profiler; the ancestor inherited the same method and
    never mentioned it.

    On low-dimensional data scikit-learn's ``algorithm="auto"`` builds a KD-tree,
    which beats any brute-force scan once the training set is large for its
    dimension (roughly ``n_samples >= 512 * 2**n_features``). Under
    ``patch_sklearn()`` such fits are handed to scikit-learn. Imported directly,
    this class always searches by brute force.

    Examples
    --------
    >>> import numpy as np
    >>> from mlxlearn.neighbors import NearestNeighbors
    >>> X = np.array([[0.0], [1.0], [2.0], [3.0]])
    >>> nn = NearestNeighbors(n_neighbors=2).fit(X)
    >>> dist, ind = nn.kneighbors([[0.5]])
    >>> ind
    array([[0, 1]])
    """

    def fit(self, X, y=None):
        """Fit the index.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : Ignored

        Returns
        -------
        self : NearestNeighbors
        """
        if self._fit_neighbors(X, None) == "sklearn":
            super().fit(X, y)
            self._set_backend("sklearn")
        return self

    def radius_neighbors(self, X=None, radius=None, return_distance=True, sort_results=False):
        """Find the neighbors within ``radius`` of each point.

        scikit-learn's implementation, run on the fitted state. See the class
        Notes for what that means for speed and precision; the parameters and
        return values are scikit-learn's.
        """
        check_is_fitted(self)
        if self._fitted_backend("radius_neighbors") == "sklearn":
            with self._inference("radius_neighbors"):
                return super().radius_neighbors(X, radius, return_distance, sort_results)

        if X is not None and not is_sparse(X):
            # The training copy is float32. scikit-learn validates the query
            # without a dtype, so a float64 query stayed float64, its Cython
            # radius search requires both sides to share a dtype, and it fell to
            # a chunked pairwise-distance path about 4.5x slower than stock.
            _reject_string_data(X)
            X = validate_data(self, X, dtype=NUMPY_FLOAT, order="C", reset=False)
        shape = self._fit_X.shape if X is None else _safe_shape(X)
        with self._inference("radius_neighbors", shape=shape):
            return super().radius_neighbors(X, radius, return_distance, sort_results)
