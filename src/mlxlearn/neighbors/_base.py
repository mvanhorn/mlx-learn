"""Shared neighbor-estimator machinery.

The three neighbor estimators differ only in what they do with the indices they
get back, so everything up to and including ``kneighbors`` lives here.

Each mlxlearn estimator subclasses its scikit-learn counterpart. That is not
laziness — it is what makes the drop-in promise real. Subclassing inherits the
exact constructor signature, ``get_params``/``set_params``, the parameter
constraints, and the estimator tags, so ``clone``, ``Pipeline``, ``GridSearchCV``
and ``isinstance`` all keep working without mlxlearn re-deriving any of it. It
also gives the patching layer an exact scikit-learn implementation to fall back
to: ``super().fit(...)`` *is* stock scikit-learn.

Layer 1 never calls that fallback. ``_mlxlearn_allow_fallback`` is ``False`` here,
so a capability mismatch raises. Layer 2 flips the flag; nothing else differs.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from numbers import Integral
from typing import ClassVar

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.neighbors import VALID_METRICS, BallTree, KDTree
from sklearn.neighbors import NearestNeighbors as _SkNearestNeighbors

from .._common.arrays import INDEX_DTYPE, NUMPY_FLOAT
from .._common.base import BackendMixin, mlx_guard
from .._common.config import get_tuning
from .._common.validation import (
    SparseInputError,
    check_finite_float32,
    check_float32_range,
    check_is_fitted,
    require_dense,
    require_supported_params,
    validate_data,
)
from .._kernels.distance import build_index
from ..exceptions import CapabilityError, UnsupportedInputError, UnsupportedParameterError

__all__ = ["MLXNeighborsMixin", "SUPPORTED_METRICS"]

#: Metrics whose squared-distance form is exactly what the blocked Gram kernel
#: computes. Anything else is a capability mismatch, not an approximation
#: opportunity: returning Euclidean results for a Manhattan request would be a
#: wrong answer delivered quickly.
SUPPORTED_METRICS = ("euclidean", "l2", "minkowski")

_SUPPORTED_ALGORITHMS = ("auto", "brute")

#: scikit-learn's ``algorithm="auto"`` never builds a tree above this many features.
_SKLEARN_TREE_MAX_FEATURES = 15

#: Set while a public inference call on a neighbor estimator has recorded its
#: diagnostics event, so that the ``kneighbors`` scikit-learn's own ``predict``
#: makes back into this class does not record a second one for the same call.
#: A context variable rather than an attribute on the estimator: it is private to
#: the thread (and task) making the call, so two threads predicting with one
#: fitted model cannot silence each other, and it holds a flag, never data.
_EVENT_RECORDED: ContextVar[bool] = ContextVar("mlxlearn_neighbors_event_recorded", default=False)


class MLXNeighborsMixin(BackendMixin):
    """``fit`` and ``kneighbors`` for the MLX brute-force neighbor path."""

    #: Neighbor search never looks at ``y``; a multi-output target changes only the
    #: host-side vote or average over the neighbors found, which is done per output
    #: exactly as scikit-learn does it. So both layers take them, and the Layer 1
    #: tags say so.
    _mlxlearn_supports_multi_output: ClassVar[bool] = True

    #: Fitted state only the MLX path creates, dropped at the start of every fit so
    #: that a refit which ends up on scikit-learn — or raises — does not keep the
    #: previous training set alive in memory and in pickles. ``_fit_X`` is here
    #: because on the MLX path it is a view of the index's host copy; scikit-learn's
    #: own fit sets it again.
    _mlxlearn_mlx_attrs: ClassVar[tuple[str, ...]] = ("_index_", "_fit_X")

    # --------------------------------------------------------------- capability

    def _check_neighbors_capability(self, X, *, data: bool = True) -> None:
        """Raise :class:`CapabilityError` for anything the MLX path cannot serve.

        Parameters
        ----------
        data : bool, default=True
            Also run the checks that scan the data, which cost a pass over ``X``.
            Patched mode skips them when it is handing a fit to scikit-learn on
            size alone: scikit-learn validates the data itself, and scanning it
            first is the kind of overhead that made patched fits slower than the
            stock ones they replaced.

        Runs before ``validate_data``, so a float64 value that overflows float32
        is reported as a capability limit — which scikit-learn's float64 path can
        serve, and therefore which patched mode can fall back on — rather than as
        scikit-learn's generic post-conversion "contains infinity" ``ValueError``.
        """
        name = type(self).__name__

        # scikit-learn's own fit also accepts a prebuilt KDTree/BallTree or another
        # fitted neighbors estimator in place of X. There is nothing for the MLX
        # path to index in one; saying so is what lets patched mode hand it over
        # instead of failing inside float32 conversion with a TypeError.
        if isinstance(X, (KDTree, BallTree, BaseEstimator)):
            raise UnsupportedInputError(
                f"{name}.fit received a prebuilt {type(X).__name__} as X. mlxlearn "
                "builds its own index from an array and cannot reuse a scikit-learn "
                "tree or fitted estimator. Pass the training array instead, or run "
                "under patch_sklearn() to have this handled by scikit-learn.",
                reason="prebuilt-index",
            )
        require_dense(X, estimator=name)

        require_supported_params(
            {"algorithm": self.algorithm, "metric": self.metric},
            {"algorithm": _SUPPORTED_ALGORITHMS, "metric": SUPPORTED_METRICS},
            estimator=name,
        )

        # 'minkowski' is only Euclidean at p == 2. The ancestor short-circuited
        # this check whenever the metric was spelled 'euclidean' or 'l2' and then
        # ignored p entirely; here p is checked against the metric that will
        # actually be used.
        if self.metric == "minkowski" and self.p != 2:
            raise UnsupportedParameterError(
                f"{name} implements metric='minkowski' only at p=2 "
                f"(Euclidean) on the MLX path; got p={self.p!r}.",
                parameter="p",
                value=self.p,
            )

        # metric_params was accepted and silently ignored by the ancestor, so a
        # weighted-Minkowski request returned unweighted Euclidean answers with no
        # warning. Wrong answers are worse than a fallback.
        if getattr(self, "metric_params", None):
            raise UnsupportedParameterError(
                f"{name} does not implement metric_params on the MLX "
                f"path; got {self.metric_params!r}. mlxlearn will not ignore it and "
                "return unweighted Euclidean distances.",
                parameter="metric_params",
                value=self.metric_params,
            )

        weights = getattr(self, "weights", "uniform")
        if callable(weights):
            raise UnsupportedParameterError(
                f"{name} does not implement a callable weights "
                "function on the MLX path.",
                parameter="weights",
                value=weights,
            )

        if data:
            check_float32_range(X)

    def _below_crossover(self, n_samples: int, n_features: int) -> tuple[bool, str]:
        """Whether stock scikit-learn is expected to win at this size.

        Uses the measured crossover from ``docs/benchmarks.md``. Handing small
        problems away is the mechanism behind "never slower than scikit-learn";
        without it, kernel-launch and transfer overhead dominates and the
        acceleration is negative.

        Small on *both* counts, rows and rows x features, so a short but wide
        training set stays on MLX. That second clause is measured too: from 40 x
        64 to 255 x 16 and 200 x 768, answering 1-256 queries, MLX ran 1.2x-2.1x
        faster than stock scikit-learn's brute force (interleaved medians).
        """
        tuning = get_tuning()
        work = n_samples * n_features
        if n_samples < tuning.knn_min_train_samples and work < tuning.knn_min_work:
            return True, (
                f"{n_samples} training samples x {n_features} features is below the "
                f"measured KNN crossover ({tuning.knn_min_train_samples} samples)"
            )
        return False, ""

    def _tree_wins(self, n_samples: int, n_features: int) -> tuple[bool, str]:
        """Whether stock scikit-learn would answer this with a KD-tree that wins.

        With ``algorithm="auto"``, scikit-learn builds a KD-tree whenever there are
        at most 15 features and ``n_neighbors < n_samples // 2``. A tree query costs
        roughly ``log n`` times something exponential in the dimension; brute force
        costs ``n * d``. So the tree wins once the data is dense enough for its
        dimension — the classic ``n >> 2**d`` condition — and loses badly to the
        GPU scan when it is not. Measured, the scan was 14x-25x slower than the
        tree at 200 000 x 4 and over 100x slower at 1 000 000 x 3, yet 2.4x-2.8x
        faster at 20 000 x 8 and 4.7x-5.3x at 200 000 x 12. A feature count alone
        cannot separate those — at 8 features the answer flips between 20 000 and
        200 000 rows — and ``knn_tree_density`` is the measured ``n / 2**d`` at
        which it flips.

        Only patched mode acts on this, because only patched mode has a tree to
        hand the work to: the Layer 1 CPU path is brute force as well, and slower
        than the GPU one.
        """
        density = get_tuning().knn_tree_density
        if not density or self.algorithm != "auto" or n_features > _SKLEARN_TREE_MAX_FEATURES:
            return False, ""
        if self.n_neighbors is not None and self.n_neighbors >= n_samples // 2:
            return False, ""
        if self._effective_metric()[0] not in VALID_METRICS["kd_tree"]:
            return False, ""
        if n_samples < density * 2**n_features:
            return False, ""
        return True, (
            f"{n_samples} training samples x {n_features} features with "
            "algorithm='auto': scikit-learn answers this with a KD-tree, which beats "
            f"brute-force search once n_samples >= {density} * 2**n_features"
        )

    def _effective_metric(self) -> tuple[str, dict]:
        """``(effective_metric_, effective_metric_params_)`` as scikit-learn sets them.

        Only meaningful for the metrics the capability check lets through:
        ``'minkowski'`` at ``p=2`` resolves to ``'euclidean'``, and ``'euclidean'``
        and ``'l2'`` keep their spelling, as they do in scikit-learn.
        """
        if self.metric == "minkowski":
            return "euclidean", {}
        return self.metric, {}

    # --------------------------------------------------------------------- fit

    def _fit_neighbors(self, X, y=None, *, target: str | None = None):
        """Validate, choose a backend, and build the index. Eager, always.

        Parameters
        ----------
        target : {"classification", "regression"}, optional
            What ``y`` is. ``None`` means unsupervised and ``y`` is ignored.
            Passed through to scikit-learn's own validation so that a continuous
            target given to a classifier, a target containing NaN, and a missing
            ``y`` all produce scikit-learn's error rather than an ``IndexError``
            from somewhere further in. The validated target is handed to
            :meth:`_fit_targets`.

        Returns
        -------
        str
            The chosen backend, or ``"sklearn"`` (Layer 2 only), in which case
            the caller must run scikit-learn's own ``fit``.
        """
        # scikit-learn validates constructor parameters in the _fit_context
        # decorator on its own fit, which overriding fit bypasses. Without this,
        # n_neighbors=0 "fitted" and predicted classes_[0] for every row, and an
        # invalid algorithm was reported as a capability mismatch instead of the
        # InvalidParameterError scikit-learn raises. LogisticRegression and SVC
        # do the same.
        self._validate_params()
        self._forget_fit()

        shape = _safe_shape(X)
        if shape is not None:
            handoff = self._sklearn_by_design(*shape)
            if handoff is not None:
                return self._hand_to_sklearn(X, shape, *handoff)

        below, detail = self._below_crossover(*shape) if shape is not None else (False, "")
        backend = self._select_backend(
            "fit",
            capability_check=lambda: self._check_neighbors_capability(X),
            below_crossover=below,
            crossover_detail=detail,
            shape=shape,
        )
        if backend == "sklearn":
            return "sklearn"

        _reject_string_data(X)
        if target is None:
            X_checked = validate_data(
                self, X, accept_sparse=False, dtype=NUMPY_FLOAT,
                ensure_min_samples=1, reset=True,
            )
            y_checked = None
        else:
            # multi_output=True, as scikit-learn validates it: a column-vector y
            # stays (n, 1), so a regressor predicts (n, 1) exactly as the stock one
            # does, instead of the shape depending on which backend fitted it.
            X_checked, y_checked = validate_data(
                self,
                X,
                y,
                accept_sparse=False,
                dtype=NUMPY_FLOAT,
                ensure_min_samples=1,
                multi_output=True,
                y_numeric=target == "regression",
                reset=True,
            )
        check_finite_float32(X_checked)

        if shape is None:
            # The input had no shape to read before validation (a list, say), so
            # the size rules run now.
            shape = X_checked.shape
            handoff = self._sklearn_by_design(*shape)
            if handoff is not None:
                return self._hand_to_sklearn(X, shape, *handoff)
            below, detail = self._below_crossover(*shape)
            if below and backend != "cpu":
                backend = self._select_backend(
                    "fit", below_crossover=True, crossover_detail=detail, shape=shape,
                )
                if backend == "sklearn":
                    return "sklearn"

        if y_checked is not None:
            self._fit_targets(y_checked)
        with mlx_guard(type(self).__name__, "fit"):
            index = build_index(X_checked, backend=backend)

        self._index_ = index
        self.n_samples_fit_ = int(X_checked.shape[0])
        self.effective_metric_, self.effective_metric_params_ = self._effective_metric()
        # scikit-learn's RadiusNeighborsMixin is inherited, so it needs the state
        # its own brute-force path reads. Populating it means radius_neighbors
        # stays correct on an MLX-fitted estimator, and it is also what lets
        # patched mode answer a sparse query with scikit-learn's brute force.
        self._fit_method = "brute"
        self._fit_X = index.X
        self._tree = None
        self._set_backend(backend)
        return backend

    def _fit_targets(self, y: np.ndarray) -> None:
        """Store the validated target. The supervised estimators implement this."""
        raise NotImplementedError

    def _forget_fit(self) -> None:
        """Drop the previous fit's backend and the MLX state it owned."""
        self._clear_backend()
        for attribute in self._mlxlearn_mlx_attrs:
            self.__dict__.pop(attribute, None)

    def _sklearn_by_design(self, n_samples: int, n_features: int) -> tuple[str, str] | None:
        """``(reason, detail)`` when patched mode hands this fit to scikit-learn by size.

        Layer 2 only, and never under ``MLXLEARN_FORCE_BACKEND``, which outranks
        every size rule. Decided from the input's shape before any pass over the
        data, so the work handed away costs what stock scikit-learn costs.
        """
        if not self._mlxlearn_allow_fallback or get_tuning().force_backend:
            return None
        below, detail = self._below_crossover(n_samples, n_features)
        if below:
            return "below-crossover", detail
        tree, detail = self._tree_wins(n_samples, n_features)
        if tree:
            return "tree-wins", detail
        return None

    def _hand_to_sklearn(self, X, shape, reason: str, detail: str) -> str:
        """Record a by-design handoff to scikit-learn and return ``"sklearn"``.

        Parameter and sparsity mismatches still win, as they do in
        ``_select_backend``: "we cannot do this at all" outranks "this is small",
        and is recorded — and warned about — as what it is.
        """
        try:
            self._check_neighbors_capability(X, data=False)
        except CapabilityError as exc:
            return self._handle_capability_error(exc, "fit", shape=shape)
        # Not a warning: dispatching this to scikit-learn is the design working.
        self._backend_event("fit", "sklearn", reason=reason, detail=detail, shape=shape, fallback=True)
        return "sklearn"

    # -------------------------------------------------------------- kneighbors

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        """Find the k nearest neighbors of each point.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features), default=None
            Query points. ``None`` means "the training data", and each point's own
            index is excluded from its neighbors.
        n_neighbors : int, default=None
            Defaults to the estimator's ``n_neighbors``.
        return_distance : bool, default=True
            Whether to return distances alongside indices.

        Returns
        -------
        neigh_dist : ndarray of shape (n_queries, n_neighbors)
            Euclidean distances, float64, ascending. Only when
            ``return_distance=True``.
        neigh_ind : ndarray of shape (n_queries, n_neighbors)
            Training indices, int64, ordered by ``(distance, index)``.

        Notes
        -----
        On an MLX-fitted estimator a sparse ``X`` raises
        :class:`~mlxlearn.exceptions.SparseInputError`. Under ``patch_sklearn()``
        it is answered by scikit-learn's brute-force search instead, which reads
        the fitted state directly — the one inference call that is not served by
        the fitted backend, and it is recorded as a fallback.
        """
        check_is_fitted(self)
        if self._fitted_backend("kneighbors") == "sklearn":
            with self._inference("kneighbors"):
                return super().kneighbors(X, n_neighbors, return_distance)
        return self._kneighbors("kneighbors", X, n_neighbors, return_distance)

    def _kneighbors(self, operation: str, X, n_neighbors=None, return_distance=True):
        """``kneighbors`` on an MLX-fitted estimator, recorded once as ``operation``.

        ``predict`` and ``predict_proba`` call this directly rather than through
        ``kneighbors``, so a public call records exactly one event, named after
        the call the user made.
        """
        n_neighbors = self._resolve_n_neighbors(n_neighbors)
        query_is_train = X is None

        if not query_is_train:
            try:
                require_dense(X, estimator=type(self).__name__, operation=operation)
            except SparseInputError as exc:
                # Layer 1 raises here. Layer 2 answers with scikit-learn's brute
                # force, which reads the state _fit_neighbors populated for it —
                # scikit-learn serves this query, so raising would break code that
                # works unpatched. fallback_policy still applies.
                self._handle_capability_error(exc, operation, shape=_safe_shape(X))
                with self._inference(operation, record=False):
                    return super().kneighbors(X, n_neighbors, return_distance)

        if query_is_train:
            Q = self._index_.X
        else:
            _reject_string_data(X)
            Q = validate_data(self, X, accept_sparse=False, dtype=NUMPY_FLOAT, reset=False)
            check_finite_float32(Q)
        k = self._check_n_neighbors_bound(n_neighbors, query_is_train, n_queries=Q.shape[0])

        with self._inference(operation, shape=Q.shape):
            with mlx_guard(type(self).__name__, operation):
                dist, idx = self._index_.query(
                    Q, k, return_distance=return_distance, exclude_self=query_is_train
                )
        return (dist, idx) if return_distance else idx

    def _resolve_n_neighbors(self, n_neighbors):
        """Validate an explicit ``n_neighbors`` with scikit-learn's exact messages.

        Only the checks scikit-learn makes before validating the query; the upper
        bound needs the query's row count and comes after.
        """
        if n_neighbors is None:
            return self.n_neighbors
        if n_neighbors <= 0:
            # int(), as scikit-learn's "%d" truncates: -1.5 is reported as -1.
            raise ValueError(f"Expected n_neighbors > 0. Got {int(n_neighbors)}")
        if not isinstance(n_neighbors, Integral):
            raise TypeError(
                f"n_neighbors does not take {type(n_neighbors)} value, "
                "enter integer value"
            )
        return n_neighbors

    def _check_n_neighbors_bound(self, n_neighbors, query_is_train: bool, *, n_queries: int) -> int:
        """The ``n_neighbors <= n_samples_fit`` check, message and all."""
        n_samples_fit = self.n_samples_fit_
        limit = n_samples_fit - int(query_is_train)
        if n_neighbors > limit:
            inequality = "n_neighbors < n_samples_fit" if query_is_train else (
                "n_neighbors <= n_samples_fit"
            )
            raise ValueError(
                f"Expected {inequality}, but n_neighbors = {n_neighbors}, "
                f"n_samples_fit = {n_samples_fit}, n_samples = {n_queries}"
            )
        return int(n_neighbors)

    # ------------------------------------------------------------------ helpers

    @contextmanager
    def _inference(
        self,
        operation: str,
        *,
        shape: tuple[int, ...] | None = None,
        record: bool = True,
    ) -> Iterator[None]:
        """Record one sticky event for a public inference call, and no nested ones.

        scikit-learn's ``predict`` calls ``self.kneighbors``, so on a model fitted
        by scikit-learn one ``predict`` used to be recorded twice, as ``predict``
        and again as ``kneighbors``, while the MLX path recorded it only as
        ``kneighbors``. The same call is now one event named after it, whichever
        backend fitted the model. ``record=False`` only silences nested calls, for
        a caller that has already recorded the decision itself.
        """
        if record and not _EVENT_RECORDED.get():
            self._note_inference(operation, shape=shape)
        token = _EVENT_RECORDED.set(True)
        try:
            yield
        finally:
            _EVENT_RECORDED.reset(token)

    def _query_indices(self, X, operation: str, *, need_distance: bool):
        """One neighbor-acquisition path for ``predict`` and ``predict_proba``.

        The ancestor had three — a prefix-cached ordered query for small k, an
        unordered query for large k, and a distance-returning query for
        ``weights='distance'`` — with nothing asserting they agreed on the
        candidate set. One path cannot disagree with itself.
        """
        dist, idx = self._kneighbors(operation, X, return_distance=True)
        return (dist if need_distance else None), idx


def _safe_shape(X) -> tuple[int, ...] | None:
    shape = getattr(X, "shape", None)
    if shape is None or len(shape) != 2:
        return None
    return (int(shape[0]), int(shape[1]))


def _reject_string_data(X) -> None:
    """Refuse string data the way scikit-learn's neighbor estimators do.

    They validate with ``dtype="numeric"``, which rejects arrays of strings and
    bytes. mlxlearn validates with an explicit float32 dtype so the conversion
    happens once, and an explicit dtype makes ``check_array`` parse ``"1.5"`` into
    1.5 instead — so the MLX path accepted input stock scikit-learn rejects, and
    under ``patch_sklearn()`` whether it did depended on the size of the data.
    Object arrays are left alone: scikit-learn converts those as well.
    """
    dtype = getattr(X, "dtype", None)
    if dtype is None and isinstance(X, (list, tuple)):
        try:
            dtype = np.asarray(X).dtype
        except ValueError:
            # Ragged input. validate_data reports it in scikit-learn's words.
            return
    if isinstance(dtype, np.dtype) and dtype.kind in "USV":
        raise ValueError(
            "dtype='numeric' is not compatible with arrays of bytes/strings."
            "Convert your data to numeric values explicitly instead."
        )


def compute_weights(dist: np.ndarray | None, weights) -> np.ndarray | None:
    """Turn neighbor distances into per-neighbor weights.

    Reimplements the ~15 lines of ``sklearn.neighbors._base._get_weights`` that
    mlxlearn needs, rather than importing a private symbol whose rename in any
    scikit-learn release would make the module unimportable.

    ``'distance'`` weighting is ``1/d``. A query that coincides exactly with a
    training point would give that neighbor infinite weight, so — matching
    scikit-learn — any row containing a zero distance is replaced by the 0/1
    indicator of its zero entries: coincident points get all the weight and every
    other neighbor gets none.
    """
    if weights in (None, "uniform"):
        return None
    if weights != "distance":
        raise ValueError(f"weights not recognized: should be 'uniform' or 'distance', got {weights!r}")
    if dist is None:
        raise ValueError("weights='distance' requires neighbor distances.")

    with np.errstate(divide="ignore"):
        w = 1.0 / np.asarray(dist, dtype=np.float64)

    coincident = np.isinf(w)
    rows_with_coincident = np.any(coincident, axis=1)
    if np.any(rows_with_coincident):
        w[rows_with_coincident] = coincident[rows_with_coincident].astype(np.float64)

    if np.any(np.all(w == 0, axis=1)):
        raise ValueError(
            "All neighbors of some sample are getting zero weights. Please modify "
            "'weights' to avoid this case if you are using a user-defined function."
        )
    return w


def _vote_totals(codes: np.ndarray, weights: np.ndarray | None, n_classes: int) -> np.ndarray:
    """Per-row total weight of each class code, shape ``(n_queries, n_classes)``, float64.

    One ``np.bincount`` over row-offset codes rather than ``np.add.at``, which is
    3-4x slower. Both accumulate each (row, class) cell in neighbor order, so the
    sums are bitwise identical.
    """
    n_queries = codes.shape[0]
    flat = (np.arange(n_queries, dtype=np.intp)[:, None] * n_classes + codes).ravel()
    totals = np.bincount(
        flat,
        weights=None if weights is None else weights.ravel(),
        minlength=n_queries * n_classes,
    )
    return totals.reshape(n_queries, n_classes).astype(np.float64, copy=False)


def weighted_mode(codes: np.ndarray, weights: np.ndarray | None, n_classes: int) -> np.ndarray:
    """Per-row weighted majority vote over integer class codes.

    Ties go to the smallest class code, matching scikit-learn. Vectorized: the
    ancestor imported ``sklearn.utils.extmath.weighted_mode`` and
    ``sklearn.utils.fixes._mode``, both private.
    """
    # argmax returns the first maximum, i.e. the smallest class code on a tie.
    return _vote_totals(codes, weights, n_classes).argmax(axis=1).astype(INDEX_DTYPE)


def class_probabilities(codes: np.ndarray, weights: np.ndarray | None, n_classes: int) -> np.ndarray:
    """Per-row normalized class scores, shape ``(n_queries, n_classes)``, float64."""
    proba = _vote_totals(codes, weights, n_classes)
    total = proba.sum(axis=1, keepdims=True)
    np.divide(proba, np.where(total > 0, total, 1.0), out=proba)
    return proba


# Re-exported so the estimator modules do not each import scikit-learn's base.
_SK_NEAREST_NEIGHBORS = _SkNearestNeighbors
