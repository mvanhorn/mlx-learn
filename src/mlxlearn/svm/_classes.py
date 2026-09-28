"""C-support-vector classification, accelerated with MLX.

The estimator subclasses :class:`sklearn.svm.SVC`. That is what makes the drop-in
promise real: the constructor signature, ``get_params``/``set_params``, the
parameter constraints and the estimator tags are inherited rather than restated,
so ``clone``, ``Pipeline``, ``GridSearchCV`` and ``isinstance`` keep working, and
``super().fit(...)`` *is* stock scikit-learn for the patching layer to fall back
to. Layer 1 never calls that fallback — ``_mlxlearn_allow_fallback`` is ``False``,
so a capability mismatch raises.

What is actually solved here is LIBSVM's C-SVC dual, exactly, by the SMO in
``_solvers/smo.py``. There is no random-feature approximation, no subsampled
training set and no ridge surrogate anywhere in this path. Those things make a
different classifier; shipping one under this class's name would mean every
accuracy number a user measured was measured on something other than what the
name says.

The three places where matching scikit-learn is fiddlier than it looks:

**The one-vs-one subproblem ordering.** LIBSVM builds the subproblem for classes
``(i, j)`` as *all class-i rows in ascending order, then all class-j rows*, labels
``+1`` then ``-1``. It is not the ascending union. Everything downstream —
``support_``'s grouping, ``dual_coef_``'s block layout, and the working-set
tie-breaks that decide which margin points end up with a nonzero α — is stated in
that ordering, so mlxlearn adopts it rather than reindexing afterwards.

**The binary sign flip.** For two classes scikit-learn negates the public
``dual_coef_`` and ``intercept_`` relative to LIBSVM's, and negates
``decision_function`` to match, so that a positive decision value means
``classes_[1]``. For three or more classes it does not. mlxlearn keeps LIBSVM's
convention internally, in ``_pair_weights_``/``_pair_rho_``, and applies the flip
only at the public boundary — one place to be right instead of several.

**``dual_coef_``'s block layout.** Pair ``(i, j)``'s weights are split across two
different *rows* of ``dual_coef_``: the class-i half lands in row ``j-1`` and the
class-j half in row ``i``. Every other entry of those two column ranges stays
exactly zero. The layout is not verifiable by looking at the array, and it is not
verified by mlxlearn's own ``decision_function`` either, which reads
``_pair_weights_`` and never ``dual_coef_`` — a scrambled layout would leave it
untouched. What checks it is evaluating the *public* state: stock LIBSVM's decision
function run on mlxlearn's fitted attributes, a decision function rebuilt from
``dual_coef_`` and ``support_vectors_`` alone, and the multiclass linear ``coef_``
scikit-learn derives from them. ``tests/parity/test_svc.py`` does all three.
"""

from __future__ import annotations

import inspect
import warnings
from typing import ClassVar, NamedTuple

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import SVC as _SkSVC
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import column_or_1d

from .._common import device as _device_state
from .._common.base import BackendMixin, mlx_guard
from .._common.config import get_config, get_tuning
from .._common.device import physical_memory_bytes
from .._common.validation import (
    FLOAT32_MAX,
    check_consistent_length,
    check_float32_range,
    check_is_fitted,
    check_sample_weight,
    compute_class_weight,
    is_sparse,
    require_dense,
    require_supported_params,
    validate_data,
)
from .._kernels.kernels import (
    SUPPORTED_KERNELS,
    KernelBasis,
    KernelSpec,
    build_kernel_rows,
    frame_spec,
    kernel_diagonal,
    kernel_frame,
    kernel_matrix,
    resolve_gamma,
)
from .._solvers.smo import SMOConfig, solve_c_svc
from ..exceptions import (
    CapabilityError,
    DeviceError,
    UnsupportedInputError,
    UnsupportedParameterError,
)

__all__ = ["SVC", "MLXSVCMixin"]

_SUPPORTED_SHAPES = ("ovo", "ovr")

#: Whether the installed scikit-learn deprecates ``probability`` (1.9+), read off the
#: inherited signature rather than a version string: the sentinel is the fact that
#: matters, and it is what scikit-learn's own fit tests.
_PROBABILITY_DEPRECATED = (
    inspect.signature(_SkSVC.__init__).parameters["probability"].default == "deprecated"
)


def _safe_shape(X) -> tuple[int, ...] | None:
    shape = getattr(X, "shape", None)
    if shape is None or len(shape) != 2:
        return None
    return (int(shape[0]), int(shape[1]))


def _check_validated_range(X: np.ndarray) -> None:
    """:func:`check_float32_range` on a validated (finite) array, cheaply.

    Two reductions rule out the common case; the full scan, and its error, run only
    when a value actually exceeds float32. Inference pays this on every call.
    """
    if X.dtype.itemsize > 4 and X.size and not max(X.max(), -X.min()) <= FLOAT32_MAX:
        check_float32_range(X)


def _libsvm_sample_weight(sample_weight, X: np.ndarray) -> np.ndarray | None:
    """Validate ``sample_weight`` with LIBSVM's meaning for a negative weight.

    scikit-learn's LIBSVM keeps only the rows with ``W > 0`` (``remove_zero_weight``
    in ``svm.cpp``), so a negative weight is *exactly* a zero weight: stock
    scikit-learn fitted with one and with the other returns bit-identical models.
    The shared validator rejects negatives because a negative weight makes a loss
    such as the logistic one non-convex; that rationale does not reach an SVC box
    constraint, and rejecting them here broke ``patch_sklearn()`` for code stock
    scikit-learn runs. So they are clipped to zero, and ``_fit_ovo`` drops them with
    every other empty box.
    """
    if sample_weight is not None:
        raw = np.asarray(sample_weight, dtype=np.float64)
        if np.any(raw < 0.0):
            # NaN compares false and survives, for the validator to reject.
            sample_weight = np.where(raw < 0.0, 0.0, raw)
    return check_sample_weight(sample_weight, X, dtype=np.float64)


def ovr_decision_function(
    predictions: np.ndarray, confidences: np.ndarray, n_classes: int
) -> np.ndarray:
    """Turn one-vs-one votes into one-vs-rest decision values.

    Reimplements the ~15 lines of scikit-learn's ``_ovr_decision_function``, which
    is private and whose rename in any release would make this module
    unimportable. The result is the vote count plus a tie-breaking term built from
    the summed pairwise confidences and scaled into ``(-1/3, 1/3)`` so that it can
    order samples inside a vote level without ever moving one across a level.

    Parameters
    ----------
    predictions : ndarray of shape (n_samples, n_pairs)
        ``0`` when the pair's first class won, ``1`` when the second did.
    confidences : ndarray of shape (n_samples, n_pairs)
        Signed pair confidences, positive for the pair's second class.
    n_classes : int

    Returns
    -------
    ndarray of shape (n_samples, n_classes), float64
    """
    n_samples = predictions.shape[0]
    votes = np.zeros((n_samples, n_classes), dtype=np.float64)
    summed = np.zeros((n_samples, n_classes), dtype=np.float64)
    k = 0
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            summed[:, i] -= confidences[:, k]
            summed[:, j] += confidences[:, k]
            votes[predictions[:, k] == 0, i] += 1
            votes[predictions[:, k] == 1, j] += 1
            k += 1
    return votes + summed / (3 * (np.abs(summed) + 1))


def ovo_vote(decision: np.ndarray, n_classes: int) -> np.ndarray:
    """Class index per row from one-vs-one decision values, LIBSVM's rule.

    Pair ``(i, j)`` votes for ``i`` when its decision value is strictly positive
    and for ``j`` otherwise — note that an exact zero votes for ``j``, which is
    what puts a sample sitting precisely on a binary decision boundary in
    ``classes_[1]``. Ties in the vote count go to the lowest class index, matching
    LIBSVM's strict ``>`` scan and :func:`numpy.argmax`.
    """
    n_samples = decision.shape[0]
    votes = np.zeros((n_samples, n_classes), dtype=np.int64)
    k = 0
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            positive = decision[:, k] > 0
            votes[positive, i] += 1
            votes[~positive, j] += 1
            k += 1
    return votes.argmax(axis=1)


class _KernelProblem(NamedTuple):
    """The kernel side of a fit, resolved and range-checked before anything is solved."""

    spec: KernelSpec  # as the user asked for it; ``gamma`` is scikit-learn's ``_gamma``
    basis: KernelBasis  # the float32 compute frame, with the frame's scale in its spec
    X32: np.ndarray  # the training matrix, mapped into that frame


class MLXSVCMixin(BackendMixin):
    """Backend selection, the one-vs-one fit, and inference for :class:`SVC`."""

    #: Fitted state only the MLX path creates. A refit that ends on scikit-learn
    #: never overwrites it, so without clearing it the previous fit's pair weights
    #: and device-resident support vectors would stay in memory and in every pickle.
    _mlxlearn_mlx_attrs: ClassVar[tuple[str, ...]] = (
        "_pair_weights_",
        "_pair_rho_",
        "_pair_kkt_violation_",
        "_kernel_spec_",
        "_kernel_center_",
        "_kernel_scale_",
        "_kernel_basis_",
    )

    def _clear_mlx_state(self) -> None:
        for attribute in self._mlxlearn_mlx_attrs:
            self.__dict__.pop(attribute, None)

    # --------------------------------------------------------------- capability

    def _check_svc_capability(self, X) -> None:
        """Raise :class:`CapabilityError` for anything the MLX path cannot serve.

        Only what is visible without scanning the data: sparsity and parameters.
        The data-dependent limits — float32 range, a ``gamma`` the compute frame
        cannot represent — are checked after validation, by :meth:`fit`, and only
        once the problem is known to be headed for MLX. Scanning an array that
        patched mode is about to hand to scikit-learn anyway is pure overhead.
        """
        name = type(self).__name__
        require_dense(X, estimator=name)

        if callable(self.kernel):
            raise UnsupportedParameterError(
                f"{name} does not implement a callable kernel on the MLX path. "
                f"Supported kernels are: {', '.join(repr(k) for k in SUPPORTED_KERNELS)}.",
                parameter="kernel",
                value=self.kernel,
            )
        require_supported_params(
            {
                "kernel": self.kernel,
                "decision_function_shape": self.decision_function_shape,
            },
            {
                "kernel": SUPPORTED_KERNELS,
                "decision_function_shape": _SUPPORTED_SHAPES,
            },
            estimator=name,
        )

        # Platt scaling needs a cross-validated inner fit and its own Newton
        # solver; 0.1.0 ships neither, and an uncalibrated sigmoid fitted on
        # in-sample margins — the shortcut that makes this look cheap — is
        # systematically overconfident. Better to name the supported alternative.
        # scikit-learn 1.9 deprecated `probability` and defaults it to the string
        # sentinel "deprecated", which is truthy. Reading it directly therefore
        # rejected every default-constructed SVC on 1.9 — a version pair
        # pyproject.toml declares supported.
        probability = getattr(self, "probability", False)
        if probability == "deprecated":
            probability = False
        if probability:
            raise UnsupportedParameterError(
                f"{name} does not implement probability=True on the MLX path: "
                "Platt calibration is not part of mlxlearn 0.1.0. Wrap the "
                "estimator in sklearn.calibration.CalibratedClassifierCV to get "
                "calibrated probabilities, or use sklearn.svm.SVC directly.",
                parameter="probability",
                value=probability,
            )

        # A non-positive numeric gamma is checked here rather than deep inside
        # kernel evaluation. Raising it from inside mlx_guard meant a gamma
        # mlxlearn cannot serve propagated straight out of Layer 2 as well, even
        # though scikit-learn accepts gamma=0.0 and would have produced a model.
        # A capability limit has to be discovered while there is still somewhere
        # to fall back to. The linear kernel never reads gamma, so any legal value
        # — scikit-learn's constraint admits only >= 0 — is served there.
        #
        # Only the numeric case is checked, and deliberately without touching X:
        # "scale" and "auto" always resolve to something positive, and reaching
        # into an unvalidated X here would pre-empt the input errors scikit-learn's
        # own validation is supposed to raise.
        if not isinstance(self.gamma, str) and not callable(self.kernel):
            try:
                gamma_value = float(self.gamma)
            except (TypeError, ValueError):
                gamma_value = None
            if gamma_value is not None and gamma_value <= 0.0 and self.kernel != "linear":
                raise UnsupportedParameterError(
                    f"{name} does not implement gamma={gamma_value!r} with "
                    f"kernel={self.kernel!r}; the MLX kernels require a strictly "
                    "positive gamma. scikit-learn accepts it, so under patch_sklearn() "
                    "this falls back rather than failing.",
                    parameter="gamma",
                    value=gamma_value,
                )

    def _below_crossover(
        self, n_samples: int, n_features: int, n_classes: int | None = None
    ) -> tuple[bool, str]:
        """Whether stock scikit-learn is expected to win at this size.

        The thresholds here are **high**, and the linear kernel is excluded
        outright, because the measurements are unflattering and pretending
        otherwise would put the slower path in front of users.

        Where the time goes, measured. Up to the row cache's capacity — about
        7 240 rows per one-vs-one subproblem at the default ``cache_size=200`` — a
        subproblem's whole Gram is built by **one** blocked MLX matmul, and the SMO
        loop then runs over it in NumPy without touching the device again. MLX's
        entire contribution is that matmul; what it loses is the loop, about
        20–40 µs of NumPy overhead per iteration against well under 1 µs for
        LIBSVM, which also shrinks its working set. Iteration counts match, so the
        loss scales with the count.

        That is why **width** decides and height does not: LIBSVM pays ``O(n·d)``
        for every kernel row it misses, and the one matmul is cheap next to that
        only when ``d`` is large, while mlxlearn's per-iteration cost does not
        depend on ``d`` at all. rbf at 4 000 × 32 measured 0.47× and the *same
        sample count* at 4 000 × 256 **2.93×**; over a 64-configuration grid at
        ``d ∈ {10, 20}`` mlxlearn was slower in 58 of 64, worst 0.013×. Hence
        both a sample floor and a work floor, with work the one that actually
        discriminates.

        The linear kernel is excluded at every size — 0.01×–0.06× throughout, and
        it degrades with C. Its many cheap iterations are exactly the case where
        per-iteration overhead is the whole cost, and no threshold fixes that.

        **Many classes.** One-vs-one solves ``K(K−1)/2`` subproblems of ``2n/K``
        rows on average, each paying its own Gram and its own loop, so a problem
        whose total clears both floors can still be a pile of tiny losses — at
        2 048 × 512, ``K=16`` measured 0.89× and ``K=128`` 0.10×. When
        ``n_classes`` is given, the *mean* pair's work ``(2n/K)·d`` must also
        clear ``svc_min_pair_work``. The mean, not the largest pair: with
        imbalanced classes one big pair can hide thousands of tiny ones, and it
        is their count that costs. For two classes the mean pair is the whole
        problem, so binary routing is unchanged.

        Above the cache's capacity the picture changes: rows are evaluated on
        demand, one device round trip per cache miss, and those round trips —
        not the NumPy loop — dominate. Nothing in the benchmark grid reaches that
        regime; it is stated so that nobody tunes this gate on the wrong cause.

        None of this is a property of the objective, which is solved correctly:
        the primal agrees with LIBSVM to 5.2e-07 relative and the KKT conditions
        hold on an independent check. See ``docs/benchmarks.md``.
        """
        tuning = get_tuning()

        if self.kernel == "linear":
            return True, (
                "the linear kernel is never faster on the MLX path (measured "
                "0.01x-0.06x across every size tested), so it is always handed to "
                "scikit-learn"
            )

        work = n_samples * n_features
        if n_samples < tuning.svc_min_samples or work < tuning.svc_min_work:
            return True, (
                f"{n_samples} samples x {n_features} features is below the measured "
                f"SVC crossover ({tuning.svc_min_samples} samples and "
                f"{tuning.svc_min_work} element-ops, both required)"
            )
        if n_classes is not None and n_classes > 2:
            pair_rows = 2.0 * n_samples / n_classes
            if pair_rows * n_features < tuning.svc_min_pair_work:
                return True, (
                    f"{n_classes} classes split {n_samples} samples into "
                    f"{n_classes * (n_classes - 1) // 2} one-vs-one subproblems of "
                    f"{pair_rows:.0f} rows on average, and {pair_rows:.0f} x "
                    f"{n_features} is below the measured per-pair crossover "
                    f"({tuning.svc_min_pair_work} element-ops)"
                )
        return False, ""

    # ------------------------------------------------------------------ targets

    def _validate_svc_targets(self, y) -> np.ndarray:
        """Set ``classes_`` and ``class_weight_``; return integer class codes.

        Reimplements scikit-learn's ``BaseSVC._validate_targets`` from public
        pieces, including its error message for a single class, so code written
        against that contract keeps working.
        """
        y_flat = column_or_1d(y, warn=True)
        check_classification_targets(y_flat)
        classes, encoded = np.unique(y_flat, return_inverse=True)
        # compute_class_weight is what rejects an unknown class_weight string and
        # a dict naming a class that is not in y. The ancestor accepted both
        # silently as all-ones, which turns a typo into a model that quietly
        # ignores the weighting it was asked for.
        self.class_weight_ = np.asarray(
            compute_class_weight(self.class_weight, classes=classes, y=y_flat),
            dtype=np.float64,
        )
        if len(classes) < 2:
            raise ValueError(
                "The number of classes has to be greater than one; got "
                f"{len(classes)} class"
            )
        self.classes_ = classes
        return np.asarray(encoded, dtype=np.intp)

    # --------------------------------------------------------------------- fit

    def _kernel_cache_rows(self, n_samples: int) -> int:
        """Kernel rows the LRU may hold for a subproblem of this size.

        ``cache_size`` is scikit-learn's knob and it is in mebibytes, so it maps
        onto the row cache directly: ``rows = bytes / (4 · n_samples)``, rows
        being float32 exactly as LIBSVM caches them.

        Only the *ceiling* comes from tuning. A user who asks for a small cache
        gets a small cache — the cache is a pure performance knob, so honoring the
        request costs nothing but time, whereas quietly overriding it downward
        would mean ``cache_size`` had no observable effect at all and upward would
        mean a request for 32 GiB on an 8 GB laptop got granted.
        """
        tuning = get_tuning()
        budget = int(float(self.cache_size) * 1024 * 1024)
        ceiling = tuning.smo_kernel_cache_max_bytes
        installed = physical_memory_bytes()
        if installed is not None:
            ceiling = min(ceiling, int(installed * tuning.smo_kernel_cache_fraction))
        # A machine that will not report its memory must still get a usable cache.
        ceiling = max(ceiling, tuning.smo_kernel_cache_min_bytes)
        rows = min(budget, ceiling) // (4 * max(n_samples, 1))
        return int(max(2, min(max(n_samples, 2), rows)))

    def _resolve_max_iter(self, n_samples: int) -> int:
        """Turn ``max_iter=-1`` into the concrete budget the solver needs.

        ``-1`` becomes upstream LIBSVM's ``max(10⁷, 100 · n_samples)``. That is a
        deliberate deviation from scikit-learn, whose bundled LIBSVM treats ``-1``
        as unlimited: mlxlearn's SMO iterates in Python at roughly 20 µs a step,
        and a problem scikit-learn needs 7·10⁸ iterations for would run for hours
        rather than stop with a warning. The ConvergenceWarning names the budget.

        ``max_iter=0`` is a legal scikit-learn value — its constraint is
        ``Interval(Integral, -1, None, closed="left")`` — and LIBSVM honors it,
        returning an unfitted-looking model with no support vectors. Only ``-1``
        means "no limit".

        The comparison here is ``>= 0`` for that reason. It was ``> 0``, which
        sent ``max_iter=0`` down the unlimited branch: the user asked for zero
        iterations and got 462, silently. Under ``patch_sklearn()`` that reached
        code expecting scikit-learn's behavior, which is exactly the class of
        quiet substitution this library exists to avoid.
        """
        if self.max_iter >= 0:
            return int(self.max_iter)
        tuning = get_tuning()
        return max(tuning.smo_min_iter_budget, tuning.smo_max_iter_factor * n_samples)

    def _prepare_kernel_problem(self, X64: np.ndarray) -> _KernelProblem:
        """Resolve the kernel and map the training data into its float32 frame.

        Everything that can make this data unservable in float32 surfaces here, as
        a :class:`CapabilityError`, before a single subproblem is solved — which is
        what lets :meth:`fit` still hand the problem to scikit-learn in patched
        mode. See :func:`~mlxlearn._kernels.kernels.kernel_frame` for the frame.
        """
        gamma = resolve_gamma(self.gamma, X64, kernel=self.kernel)
        spec = KernelSpec(
            kernel=self.kernel,
            gamma=gamma,
            degree=int(self.degree),
            coef0=float(self.coef0),
        )
        center, scale = kernel_frame(spec, X64)
        basis = KernelBasis(frame_spec(spec, scale), center=center, scale=scale)
        return _KernelProblem(spec=spec, basis=basis, X32=basis.prepare(X64))

    def _solve_pair(self, X_pair, y_pair, C_pair, spec, center, backend):
        """Solve one one-vs-one subproblem.

        The full Gram is precomputed only when the row cache was already sized to
        hold every row — at which point the memory has been accounted for and one
        blocked matmul beats ``n`` separate row evaluations. Otherwise rows are
        evaluated on demand and the LRU bounds what is resident, so a subproblem
        larger than the cache never needs an ``n × n`` matrix to exist.

        ``_fit_ovo`` passes rows already mapped into the compute frame, with the
        frame folded into ``spec`` and no ``center``; the diagonal is then the
        closed form on exactly the float32 values the rows are built from.
        """
        n_pair = int(X_pair.shape[0])
        cache_rows = self._kernel_cache_rows(n_pair)
        k_diag = kernel_diagonal(spec, X_pair)

        if cache_rows >= n_pair:
            gram = kernel_matrix(spec, X_pair, backend=backend, center=center)
            # Rows of a materialized Gram are views, so the LRU adds no memory here.
            fetch = gram.__getitem__
        else:
            rows = build_kernel_rows(X_pair, spec, backend=backend, center=center)
            fetch = rows.row

        config = SMOConfig(
            tol=float(self.tol),
            max_iter=self._resolve_max_iter(n_pair),
            cache_rows=max(2, min(cache_rows, n_pair)),
        )
        return solve_c_svc(y_pair, C_pair, k_diag, fetch, config)

    def _fit_ovo(self, X64, y_encoded, sample_weight, backend, problem=None) -> None:
        """Fit every one-vs-one subproblem and assemble LIBSVM's model layout."""
        n_samples, n_features = X64.shape
        n_classes = int(len(self.classes_))
        if problem is None:
            problem = self._prepare_kernel_problem(X64)
        X32 = problem.X32
        compute_spec = problem.basis.spec

        # C_t = C * class_weight[class(t)] * sample_weight[t], which is exactly
        # how scikit-learn's LIBSVM builds its per-sample box.
        C_all = float(self.C) * self.class_weight_[y_encoded]
        if sample_weight is not None:
            C_all = C_all * sample_weight

        # A sample whose box is [0, 0] can never leave the lower bound, but it is
        # still a member of I_low, so its gradient keeps feeding the stopping
        # criterion and the solver runs to max_iter without ever being able to fix
        # the violation it is reporting -- measured: 10^7 iterations and a model
        # 16% worse than scikit-learn's. scikit-learn's LIBSVM drops rows whose
        # weight is not positive before solving, so mlxlearn drops them too, on
        # the effective box rather than on the raw weight so that
        # class_weight={c: 0} is handled the same way.
        active = C_all > 0.0
        if not active.all():
            empty = [
                self.classes_[c]
                for c in range(n_classes)
                if not active[y_encoded == c].any()
            ]
            if empty:
                raise ValueError(
                    "Every sample of class "
                    f"{empty[0]!r} has an effective weight of zero, so that class "
                    "cannot be fitted. Remove it from y or give it a positive "
                    "sample_weight or class_weight."
                )

        class_rows = [np.flatnonzero((y_encoded == c) & active) for c in range(n_classes)]
        pairs = [(i, j) for i in range(n_classes) for j in range(i + 1, n_classes)]
        n_pairs = len(pairs)

        pair_rows: list[np.ndarray] = []
        pair_coef: list[np.ndarray] = []
        rho = np.empty(n_pairs, dtype=np.float64)
        n_iter = np.empty(n_pairs, dtype=np.int32)
        violation = np.empty(n_pairs, dtype=np.float64)
        fit_status = 0
        exhausted_budget = 0

        for k, (i, j) in enumerate(pairs):
            rows_i = class_rows[i]
            rows_j = class_rows[j]
            index = np.concatenate([rows_i, rows_j])
            y_pair = np.concatenate(
                [np.ones(rows_i.shape[0]), -np.ones(rows_j.shape[0])]
            )
            state = self._solve_pair(
                np.ascontiguousarray(X32[index]),
                y_pair,
                C_all[index],
                compute_spec,
                None,
                backend,
            )
            pair_rows.append(index)
            # LIBSVM folds the label into the dual coefficient before storing it.
            pair_coef.append(state.alpha * y_pair)
            rho[k] = state.rho
            n_iter[k] = state.n_iter
            violation[k] = state.max_kkt_violation
            if not state.converged:
                fit_status = 1
                exhausted_budget = max(exhausted_budget, self._resolve_max_iter(index.shape[0]))

        # A row is a support vector if ANY pair gave it a nonzero coefficient, and
        # LIBSVM's test is exactly alpha != 0 -- not a threshold. The box branches
        # clip to 0 exactly, so the comparison is meaningful.
        is_support = np.zeros(n_samples, dtype=bool)
        for index, coef in zip(pair_rows, pair_coef, strict=True):
            is_support[index] |= np.abs(coef) > 0.0

        support_parts = [rows[is_support[rows]] for rows in class_rows]
        n_support = np.array([part.shape[0] for part in support_parts], dtype=np.int32)
        support = (
            np.concatenate(support_parts).astype(np.int32)
            if n_support.sum()
            else np.empty(0, dtype=np.int32)
        )
        offsets = np.concatenate([[0], np.cumsum(n_support)]).astype(np.intp)
        n_sv = int(n_support.sum())

        pair_weights = np.zeros((n_sv, n_pairs), dtype=np.float64)
        for k, (i, j) in enumerate(pairs):
            coef = pair_coef[k]
            count_i = class_rows[i].shape[0]
            pair_weights[offsets[i] : offsets[i + 1], k] = coef[:count_i][
                is_support[class_rows[i]]
            ]
            pair_weights[offsets[j] : offsets[j + 1], k] = coef[count_i:][
                is_support[class_rows[j]]
            ]

        dual_coef = np.zeros((max(n_classes - 1, 1), n_sv), dtype=np.float64)
        for k, (i, j) in enumerate(pairs):
            dual_coef[j - 1, offsets[i] : offsets[i + 1]] = pair_weights[
                offsets[i] : offsets[i + 1], k
            ]
            dual_coef[i, offsets[j] : offsets[j + 1]] = pair_weights[
                offsets[j] : offsets[j + 1], k
            ]

        # scikit-learn's own post-fit check, with its message verbatim so code that
        # matches on it keeps working. Without it a kernel that overflowed float32
        # produced NaN gradients, the solver found no eligible working pair and
        # reported convergence after one iteration, and the fit returned a NaN
        # intercept with fit_status_=0 -- a model predicting one class for
        # everything, silently. Checked before any attribute is assigned, so a
        # failed fit leaves no half-populated estimator behind.
        if not (np.isfinite(rho).all() and np.isfinite(dual_coef).all()):
            raise ValueError(
                "The dual coefficients or intercepts are not finite. The input data "
                "may contain large values and need to be preprocessed. (mlxlearn "
                "evaluates the kernel in float32, where these kernel values "
                "overflowed.)"
            )

        self.support_ = support
        self.support_vectors_ = X64[support] if n_sv else np.empty((0, n_features))
        self._n_support = n_support
        self.shape_fit_ = (int(n_samples), int(n_features))
        self.fit_status_ = fit_status
        self.n_iter_ = n_iter
        self._num_iter = n_iter
        self._gamma = problem.spec.gamma
        self._sparse = False
        self._probA = np.empty(0)
        self._probB = np.empty(0)

        # LIBSVM's convention throughout the internals: decision = K @ w - rho.
        self._pair_weights_ = pair_weights
        self._pair_rho_ = rho
        self._pair_kkt_violation_ = violation
        self._kernel_spec_ = problem.spec
        self._kernel_center_ = problem.basis.center
        self._kernel_scale_ = problem.basis.scale
        # Owns the support vectors' device copy for inference. Created here, not on
        # the first predict, so predict never changes the estimator's __dict__; the
        # device tensors inside it are built lazily and never pickled.
        self._kernel_basis_ = problem.basis

        self._dual_coef_ = dual_coef
        self._intercept_ = -rho
        if n_classes == 2:
            self.dual_coef_ = -dual_coef
            self.intercept_ = rho.copy()
        else:
            self.dual_coef_ = dual_coef
            self.intercept_ = -rho

        if fit_status:
            if self.max_iter == -1:
                limit = (
                    f"max_iter=-1, which mlxlearn resolves to a hard budget of "
                    f"{exhausted_budget} iterations per one-vs-one subproblem; unlike "
                    "scikit-learn's bundled LIBSVM it does not treat -1 as unlimited"
                )
                advice = "Pass a larger explicit max_iter to keep iterating, or consider"
            else:
                limit = f"max_iter={self.max_iter}"
                advice = "Consider"
            warnings.warn(
                f"Solver terminated early ({limit}). {advice} pre-processing your "
                "data with StandardScaler or MinMaxScaler.",
                ConvergenceWarning,
                stacklevel=2,
            )

    def _warn_sklearn_deprecations(self) -> None:
        """Emit the FutureWarnings scikit-learn's own ``fit`` emits for these params.

        Called only on the MLX path — scikit-learn's fallback warns for itself —
        so a user gets the same advance notice of a removal whichever path ran.
        Layer 1 otherwise never showed it at all.
        """
        if _PROBABILITY_DEPRECATED and self.probability != "deprecated":
            warnings.warn(
                "The `probability` parameter was deprecated in 1.9 and will be removed "
                "in version 1.11. Use `CalibratedClassifierCV(SVC(), ensemble=False)` "
                "instead of `SVC(probability=True)`",
                FutureWarning,
                stacklevel=3,
            )

    # --------------------------------------------------------------- inference

    def _inference_backend(self, operation: str, shape: tuple[int, ...] | None) -> str:
        """The fitted backend, or the MLX CPU device when the fitted GPU is gone.

        Inference stays on the backend ``fit`` chose, with one exception that does
        not re-decide anything: a model fitted on the MLX GPU and loaded where no
        GPU is usable runs the *same* MLX code on the MLX CPU device. Its state is
        backend-independent, so this is not the scikit-learn hop stickiness exists
        to prevent. It happens under ``device="auto"`` and ``"cpu"`` only; an
        explicit ``device="gpu"`` still raises, because a user who named a device
        wants that device.
        """
        backend = self._fitted_backend(operation)
        if backend == "mlx" and not _device_state.gpu_available():
            if get_config().device == "gpu":
                raise DeviceError(
                    f"{type(self).__name__} was fitted on the MLX GPU backend and no "
                    "usable MLX GPU exists in this process. device='gpu' names that "
                    "device explicitly, so mlxlearn will not compute elsewhere; set "
                    "device='auto' or 'cpu' to evaluate this model on the MLX CPU "
                    "device instead."
                )
            self._backend_event(operation, "cpu", reason="gpu-unavailable", shape=shape)
            return "cpu"
        self._note_inference(operation, shape=shape)
        return backend

    def _kernel_basis(self) -> KernelBasis:
        basis = getattr(self, "_kernel_basis_", None)
        if basis is not None:
            return basis
        # A model pickled by mlxlearn 0.1.0a2, which predates the compute frame. Its
        # centre and unscaled spec still describe a valid frame exactly.
        return KernelBasis(self._kernel_spec_, center=self._kernel_center_, scale=1.0)

    def _pairwise_decision(self, X, operation: str = "decision_function") -> np.ndarray:
        """One-vs-one decision values in LIBSVM's convention, ``(n, n_pairs)``.

        Positive means the pair's *first* class. This is the single quantity every
        public inference method is derived from, so ``predict`` and
        ``decision_function`` cannot disagree about which side of a boundary a
        sample is on.

        Queries are range-checked before they reach float32, as ``fit`` checks its
        input. Unchecked, a 1e39 query became ``inf``, the decision values NaN,
        and ``ovo_vote`` — for which ``NaN > 0`` is false — silently flipped the
        label. Inference cannot fall back (the backend is sticky), so this raises
        :class:`~mlxlearn.exceptions.UnsupportedInputError` in both layers.
        """
        if is_sparse(X):
            # scikit-learn's own error for a dense-fitted model, class and message.
            raise ValueError(
                f"cannot use sparse input in {type(self).__name__!r} trained on dense data"
            )
        # float32 queries are kept as float32: the frame is applied in float64 block
        # by block anyway, and float32 -> float64 -> float32 is exact.
        Xv = validate_data(
            self,
            X,
            accept_sparse=False,
            dtype=(np.float64, np.float32),
            order="C",
            reset=False,
        )
        backend = self._inference_backend(operation, _safe_shape(Xv))
        n_pairs = self._pair_weights_.shape[1]
        if Xv.shape[0] == 0:
            return np.empty((0, n_pairs), dtype=np.float64)
        _check_validated_range(Xv)
        with mlx_guard(type(self).__name__, operation):
            decision = self._kernel_basis().combine(
                Xv, self.support_vectors_, self._pair_weights_, backend=backend
            )
        decision -= self._pair_rho_
        if not np.isfinite(decision).all():
            raise UnsupportedInputError(
                "The decision values for these queries are not finite: the "
                f"{self.kernel!r} kernel overflowed float32, where mlxlearn evaluates "
                "it. Rescale the queries (or the training data) so the kernel values "
                "stay below ~3.4e38.",
                reason="float32-overflow",
            )
        return decision


class SVC(MLXSVCMixin, _SkSVC):
    """C-support-vector classification, accelerated with MLX.

    A drop-in replacement for :class:`sklearn.svm.SVC` for dense data and the four
    built-in kernels. The dual is solved exactly, by LIBSVM's sequential minimal
    optimization with second-order working-set selection — the same objective, the
    same stopping criterion and the same per-sample box that scikit-learn solves.

    Parameters
    ----------
    C : float, default=1.0
        Regularization parameter, the upper bound on every dual variable before
        class and sample weights scale it.
    kernel : {"linear", "poly", "rbf", "sigmoid"}, default="rbf"
        ``"precomputed"`` and callable kernels are not implemented on the MLX
        path and raise :class:`~mlxlearn.exceptions.UnsupportedParameterError`.
    degree : int, default=3
        Degree of the polynomial kernel. Evaluated by repeated squaring, so a
        negative base raised to an odd degree gives a negative value rather than
        NaN.
    gamma : {"scale", "auto"} or float, default="scale"
        ``"scale"`` is ``1 / (n_features * X.var())``, ``"auto"`` is
        ``1 / n_features``, both resolved exactly as scikit-learn resolves them —
        including its fallback to ``1.0`` only for an exactly-zero variance. The
        linear kernel does not read it, so ``gamma=0.0`` is accepted there.
    coef0 : float, default=0.0
    shrinking : bool, default=True
        **Accepted and ignored.** Shrinking is a pure speed heuristic that
        provably does not change the optimum; mlxlearn 0.1.0 does not implement
        it, and passing ``shrinking=False`` therefore also changes nothing. It is
        not raised on because doing so would reject the scikit-learn default.
    probability : bool, default=False
        ``True`` is not implemented; see Notes. On scikit-learn 1.9, where the
        parameter is deprecated and defaults to the ``"deprecated"`` sentinel,
        passing it explicitly emits scikit-learn's FutureWarning on the MLX path
        too.
    tol : float, default=1e-3
        Stopping tolerance on the maximal KKT violation.
    cache_size : float, default=200
        Kernel cache in MiB. Mapped onto the solver's LRU row cache:
        ``rows = cache_size · 2²⁰ / (4 · n_samples)``, clamped by installed memory.

        **Unlike scikit-learn, this is not purely a performance knob.** When the
        budget is large enough to hold every row, mlxlearn materializes the whole
        Gram in one blocked matmul instead of evaluating rows on demand, and the
        two accumulate in different orders. Measured on 2 000 × 10: decision
        values shift by 4.5e-04 and the iteration count changes from 801 to 794
        as the budget crosses that boundary. Both results are KKT-optimal and the
        shift is well inside the tolerance in ``docs/fp32_policy.md``, but
        scikit-learn is bit-identical across ``cache_size`` and mlxlearn is not.
        Pin ``cache_size`` explicitly if you need bit-reproducible fits.
    class_weight : dict or "balanced", default=None
    verbose : bool, default=False
        Accepted and ignored.
    max_iter : int, default=-1
        ``-1`` resolves to upstream LIBSVM's budget, ``max(10⁷, 100 · n)`` per
        one-vs-one subproblem of ``n`` rows. **This deviates from scikit-learn**,
        whose bundled LIBSVM treats ``-1`` as unlimited: mlxlearn's SMO iterates in
        Python, and uncapped it would turn a fit scikit-learn finishes in a minute
        into hours. A fit that stops on the budget sets ``fit_status_=1`` and names
        the budget in its ConvergenceWarning; pass a larger explicit ``max_iter``
        to keep going.
    decision_function_shape : {"ovo", "ovr"}, default="ovr"
        Read at call time, not frozen at fit time, matching scikit-learn. Ignored
        for two classes, which always return a 1-D decision function.
    break_ties : bool, default=False
    random_state : int, RandomState or None, default=None
        Unused: this solver is deterministic and 0.1.0 has no probability path.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
    support_ : ndarray of shape (n_SV,), int32
        Training indices of the support vectors, grouped by class and ascending
        within each class.
    support_vectors_ : ndarray of shape (n_SV, n_features), float64
    n_support_ : ndarray of shape (n_classes,), int32
    dual_coef_ : ndarray of shape (n_classes - 1, n_SV), float64
        LIBSVM's one-vs-one block layout, negated for the binary case exactly as
        scikit-learn negates it.
    intercept_ : ndarray of shape (n_classes * (n_classes - 1) / 2,), float64
    class_weight_ : ndarray of shape (n_classes,), float64
    n_iter_ : ndarray of shape (n_pairs,), int32
        SMO updates performed per one-vs-one subproblem.
    fit_status_ : int
        ``0`` if every one-vs-one subproblem converged, ``1`` if any hit
        ``max_iter`` — which, as in scikit-learn, also raises a
        :class:`~sklearn.exceptions.ConvergenceWarning`.
    shape_fit_ : tuple of int
    n_features_in_ : int
    execution_backend_ : {"mlx", "cpu"}

    Notes
    -----
    **Not implemented in 0.1.0.** ``kernel="precomputed"``, callable kernels,
    sparse ``X``, and ``probability=True``. Each raises a
    :class:`~mlxlearn.exceptions.UnsupportedParameterError` or
    :class:`~mlxlearn.exceptions.SparseInputError` naming what is missing; none of
    them silently computes something else. For probabilities, wrap this estimator
    in :class:`sklearn.calibration.CalibratedClassifierCV`, which cross-validates
    the calibration instead of fitting a sigmoid on in-sample margins.

    **Precision.** Kernel entries are float32 and the dual variables, gradients
    and working-set arithmetic are float64 — the same split LIBSVM uses, which
    caches kernel rows as C ``float``. Decision values contract the float32 kernel
    against float64 coefficients in float64, so the only float32 error that
    reaches the output is the kernel's own rounding. Before anything is cast, the
    data is mapped exactly in float64: translated by the training mean (rbf only)
    and scaled by a power of two that is folded back into ``gamma``. A large
    offset — timestamps, projected coordinates — or a magnitude around 1e±19
    therefore never reaches float32 at all. See ``docs/fp32_policy.md``.

    What float32 cannot represent even so — a value past ``3.4e38``, a
    linear-kernel inner product that large, a ``gamma`` that cannot be folded
    into range — raises :class:`~mlxlearn.exceptions.UnsupportedInputError` or
    :class:`~mlxlearn.exceptions.UnsupportedParameterError` at ``fit``, which
    patched mode hands to scikit-learn, and at ``predict``, which it cannot: the
    fitted backend is sticky, and a NaN decision would silently flip the label.

    **Zero and negative weights, and where ``support_`` then differs.** A sample
    whose effective ``C`` is not positive — from a zero or negative
    ``sample_weight``, or a zero ``class_weight`` — is dropped from the
    optimization, which is what scikit-learn's LIBSVM does and produces the same
    model. scikit-learn then reports ``support_`` as indices into the *filtered*
    problem, so ``support_vectors_ != X[support_]`` and the indices point at
    unrelated rows; mlxlearn reports indices into the training data the caller
    passed, so ``support_vectors_ == X[support_]`` always holds. This is the one
    fitted attribute where mlxlearn deliberately does not reproduce
    scikit-learn's value, and it is deliberate because the alternative is
    returning indices that mean nothing.

    Examples
    --------
    >>> import numpy as np
    >>> from mlxlearn.svm import SVC
    >>> X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    >>> y = np.array([0, 0, 1, 1])
    >>> clf = SVC(kernel="linear").fit(X, y)
    >>> clf.predict([[-1.5], [1.5]])
    array([0, 1])
    >>> clf.n_support_
    array([1, 1], dtype=int32)
    >>> clf.decision_function([[1.5]]) > 0
    array([ True])
    """

    def fit(self, X, y, sample_weight=None):
        """Fit the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        sample_weight : array-like of shape (n_samples,), default=None
            Per-sample multipliers on ``C``. Must be finite. A zero or negative
            weight drops the sample, exactly as scikit-learn's LIBSVM does.

        Returns
        -------
        self : SVC
        """
        # Inherited fit is decorated with scikit-learn's validation context;
        # overriding it means the constraint check has to be requested explicitly,
        # or C=-1 would reach the solver instead of raising InvalidParameterError.
        self._validate_params()
        self._clear_backend()
        self._clear_mlx_state()
        shape = _safe_shape(X)

        # Patched mode decides from the raw shape first. Most of what it sees is
        # below the crossover, and everything below would otherwise be validated,
        # range-scanned and target-checked here only for scikit-learn to do it all
        # again: measured 1.4x-1.6x stock at 200 x 5. Only the parameter and
        # sparsity checks run first, so a capability mismatch is still reported as
        # one. With MLXLEARN_FORCE_BACKEND set the crossover does not apply, and a
        # below-crossover selection in Layer 2 always resolves to scikit-learn.
        forced = bool(get_tuning().force_backend)
        if self._mlxlearn_allow_fallback and shape is not None and not forced:
            below, detail = self._below_crossover(*shape)
            if below:
                self._select_backend(
                    "fit",
                    capability_check=lambda: self._check_svc_capability(X),
                    below_crossover=True,
                    crossover_detail=detail,
                    shape=shape,
                )
                return self._fit_with_sklearn(X, y, sample_weight)

        backend = self._select_backend(
            "fit",
            capability_check=lambda: self._check_svc_capability(X),
            shape=shape,
        )
        if backend == "sklearn":
            return self._fit_with_sklearn(X, y, sample_weight)

        X64 = validate_data(
            self,
            X,
            accept_sparse=False,
            dtype=np.float64,
            order="C",
            ensure_min_samples=1,
            reset=True,
        )
        y_encoded = self._validate_svc_targets(y)
        check_consistent_length(X64, y_encoded)
        weights = _libsvm_sample_weight(sample_weight, X64)

        below, detail = self._below_crossover(*X64.shape, n_classes=len(self.classes_))
        # Layer 1 below the crossover runs on the MLX CPU device, and needs no second
        # selection when it is already there. Layer 2 hands it to scikit-learn
        # whatever device is configured: device="cpu" selects MLX's CPU, which is
        # the same float32 path and no faster, not scikit-learn's.
        if below and not forced and (self._mlxlearn_allow_fallback or backend != "cpu"):
            backend = self._select_backend(
                "fit",
                below_crossover=True,
                crossover_detail=detail,
                shape=X64.shape,
            )
            if backend == "sklearn":
                return self._fit_with_sklearn(X, y, sample_weight)

        # The data-dependent capability limits, checked only now that MLX will run.
        # Raised this late they bypass _select_backend, so they are routed through
        # the same policy by hand: Layer 1 raises, Layer 2 falls back or raises
        # under fallback_policy="raise". The float32 range check runs on the validated
        # float64 array, which also covers object-dtype input the raw scan skipped.
        try:
            _check_validated_range(X64)
            problem = self._prepare_kernel_problem(X64)
        except CapabilityError as exc:
            self._handle_capability_error(exc, "fit", shape=X64.shape)
            return self._fit_with_sklearn(X, y, sample_weight)

        self._warn_sklearn_deprecations()
        with mlx_guard(type(self).__name__, "fit"):
            self._fit_ovo(X64, y_encoded, weights, backend, problem=problem)
        self._set_backend(backend)
        return self

    def _fit_with_sklearn(self, X, y, sample_weight):
        super().fit(X, y, sample_weight)
        self._set_backend("sklearn")
        return self

    def decision_function(self, X):
        """Signed distance of each sample to the separating hyperplanes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        dec : ndarray, float64
            ``(n_samples,)`` for two classes, where positive means
            ``classes_[1]``. Otherwise ``(n_samples, n_classes)`` when
            ``decision_function_shape="ovr"`` and
            ``(n_samples, n_classes * (n_classes - 1) / 2)`` when it is ``"ovo"``.

        Notes
        -----
        ``decision_function_shape`` is read here rather than frozen at fit time,
        so changing it after fitting changes the output shape. That matches
        scikit-learn.

        The two-class case returns 1-D regardless of ``decision_function_shape``.
        """
        check_is_fitted(self)
        if self._fitted_backend("decision_function") == "sklearn":
            self._note_inference("decision_function")
            return super().decision_function(X)

        decision = self._pairwise_decision(X)
        n_classes = len(self.classes_)
        if n_classes == 2:
            # scikit-learn's public binary convention is the negation of
            # LIBSVM's, so that a positive value means classes_[1].
            return -decision[:, 0]
        if self.decision_function_shape == "ovr":
            return ovr_decision_function(decision < 0, -decision, n_classes)
        return decision

    def predict(self, X):
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y : ndarray of shape (n_samples,)
            Labels drawn from ``classes_``.

        Notes
        -----
        Multiclass prediction is the one-vs-one vote, with ties going to the
        lowest class index. ``break_ties=True`` replaces the vote with the argmax
        of the one-vs-rest decision function, and — as in scikit-learn — is only
        meaningful with ``decision_function_shape="ovr"``.
        """
        check_is_fitted(self)
        if self._fitted_backend("predict") == "sklearn":
            self._note_inference("predict")
            return super().predict(X)

        if self.break_ties and self.decision_function_shape == "ovo":
            raise ValueError(
                "break_ties must be False when decision_function_shape is 'ovo'"
            )
        n_classes = len(self.classes_)
        if self.break_ties and self.decision_function_shape == "ovr" and n_classes > 2:
            indices = np.argmax(self.decision_function(X), axis=1)
        else:
            indices = ovo_vote(self._pairwise_decision(X, "predict"), n_classes)
        return self.classes_.take(np.asarray(indices, dtype=np.intp))
