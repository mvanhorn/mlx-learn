"""Logistic regression, accelerated with MLX.

Subclassing :class:`sklearn.linear_model.LogisticRegression` is what makes the
drop-in promise real: the constructor signature, ``get_params``/``set_params``, the
parameter constraints and the estimator tags are all inherited, so ``clone``,
``Pipeline``, ``GridSearchCV`` and ``isinstance`` keep working, and
``super().fit(...)`` *is* stock scikit-learn for the patching layer to fall back to.

Scope for 0.1.0 is one solver — L-BFGS on an L2-penalized (or unpenalized)
objective — and everything outside it raises a
:class:`~mlxlearn.exceptions.CapabilityError` naming the parameter. That is
narrower than the ancestor, which accepted ``multi_class='ovr'`` and silently fit a
multinomial model instead: a wrong answer delivered quickly is worse than a
fallback.
"""

from __future__ import annotations

import inspect
import math
import re
import sys
import warnings
from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression as _SkLogisticRegression

from .._common.base import BackendMixin, mlx_guard
from .._common.config import get_config, get_tuning
from .._common.device import inference_device
from .._common.validation import (
    check_classification_targets,
    check_float32_range,
    check_is_fitted,
    check_sample_weight,
    compute_class_weight,
    require_dense,
    require_nonnegative_weights,
    require_supported_params,
    type_of_target,
    validate_data,
)
from .._solvers.logistic import LogisticConfig, solve_logistic
from ..exceptions import (
    BackendExecutionError,
    UnsupportedInputError,
    UnsupportedParameterError,
)

try:  # scikit-learn >= 1.9: tears callbacks down after fit, whatever the outcome
    from sklearn.callback import with_callbacks as _with_callbacks

    #: The frame the decorator adds between the caller and ``fit``, which the
    #: ``stacklevel`` of a warning raised from inside ``fit`` has to step over.
    _WRAPPER_FRAMES = 1
except ImportError:  # scikit-learn 1.7 / 1.8 have no callback API

    def _with_callbacks(method):
        return method

    _WRAPPER_FRAMES = 0


__all__ = ["LogisticRegression", "MLXLogisticMixin"]

_SUPPORTED_SOLVERS = ("lbfgs",)
_SUPPORTED_PENALTIES = ("l2", None)
#: ``"deprecated"`` is scikit-learn 1.7's default sentinel and ``"auto"`` is the
#: value it replaced; both mean "binary problems get one weight vector, everything
#: else gets the multinomial loss", which is what this implementation does.
_SUPPORTED_MULTI_CLASS = ("deprecated", "auto")

#: ``(major, minor)`` of the installed scikit-learn. Only the fit-time warnings are
#: version-gated on it; capability is decided on resolved values (see
#: :meth:`MLXLogisticMixin._effective_penalty`), never on version numbers.
_SKLEARN = tuple(int(part) for part in re.match(r"(\d+)\.(\d+)", sklearn.__version__).groups())

#: Whether ``penalty`` is a deprecated sentinel on the installed scikit-learn (1.8+).
_PENALTY_DEPRECATED = (
    inspect.signature(_SkLogisticRegression.__init__).parameters["penalty"].default
    == "deprecated"
)

#: Identities of the estimators whose public inference call is in progress on this
#: thread or task. scikit-learn's ``predict``, ``predict_proba``, ``predict_log_proba``
#: and ``score`` call back into the overridden ``decision_function`` /
#: ``predict_proba`` / ``predict``, so without it one public call recorded two or
#: three diagnostics events, labelled with whichever method ran last. A
#: ``ContextVar`` holding an immutable set is per-thread and per-task by
#: construction, and an identity is only in it for the duration of the call that
#: put it there — nothing here outlives a call or is keyed on data.
_INFERENCE_IN_PROGRESS: ContextVar[frozenset[int]] = ContextVar(
    "mlxlearn_logistic_inference", default=frozenset()
)


class MLXLogisticMixin(BackendMixin):
    """Capability gating and crossover dispatch for the logistic estimators."""

    # ------------------------------------------------------------ version shims

    def _effective_penalty(self):
        """Resolve the penalty the way the installed scikit-learn resolves it.

        scikit-learn deprecated ``penalty`` in 1.8 in favor of ``l1_ratio``, with
        removal in 1.10, so the two spellings coexist across the range
        ``pyproject.toml`` declares:

        * **1.7** — ``penalty`` is real, default ``"l2"``; ``l1_ratio`` defaults
          to ``None``.
        * **1.8+** — ``penalty`` defaults to the sentinel ``"deprecated"`` and the
          effective penalty comes from ``l1_ratio`` (0 → l2, 1 → l1, else
          elasticnet), with ``C=inf`` meaning unpenalized. ``l1_ratio`` defaults
          to ``0.0``.

        Reading the raw attributes broke on both sides of that change: ``penalty``
        read as the literal string ``"deprecated"`` and was rejected as
        unsupported, and the old ``l1_ratio is not None`` guard rejected 1.9's
        own default. Capability is decided on the *resolved* value instead, so
        the same request means the same thing on every supported version.
        """
        penalty = getattr(self, "penalty", "deprecated")
        if penalty != "deprecated":
            return penalty

        l1_ratio = getattr(self, "l1_ratio", None)
        if l1_ratio is None or l1_ratio == 0:
            resolved = "l2"
        elif l1_ratio == 1:
            resolved = "l1"
        else:
            resolved = "elasticnet"
        return None if self.C == math.inf else resolved

    def _mirror_fit_warnings(self) -> None:
        """Emit the warnings scikit-learn's own ``fit`` would have emitted.

        scikit-learn raises its deprecation notices for ``penalty``,
        ``l1_ratio=None`` and ``n_jobs`` from inside ``fit``, so a fit that never
        reaches ``super().fit`` never showed them — and a Layer 1 estimator never
        reaches it at all. The only advance notice before 1.10 turns these into
        errors depended on which backend happened to run. Called once, on the MLX
        path only, so a scikit-learn fallback does not warn twice.

        The conditions and wording follow the installed version's ``fit``;
        ``tests/parity/test_logistic.py`` compares them against scikit-learn's own
        on every version CI runs.
        """
        penalty, l1_ratio, C = self.penalty, self.l1_ratio, self.C
        emitted: list[tuple[str, type[Warning]]] = []

        if not _PENALTY_DEPRECATED:  # scikit-learn 1.7
            if penalty != "elasticnet" and l1_ratio is not None:
                emitted.append((
                    "l1_ratio parameter is only used when penalty is 'elasticnet'. "
                    f"Got (penalty={penalty})",
                    UserWarning,
                ))
            if penalty is None and C != 1.0:
                emitted.append((
                    "Setting penalty=None will ignore the C and l1_ratio parameters",
                    UserWarning,
                ))
            if getattr(self, "multi_class", "deprecated") == "auto":
                emitted.append((
                    "'multi_class' was deprecated in version 1.5 and will be removed in "
                    "1.8. From then on, it will always use 'multinomial'. Leave it to "
                    "its default value to avoid this warning.",
                    FutureWarning,
                ))
        else:  # scikit-learn 1.8, 1.9
            resolved = self._effective_penalty()
            if penalty == "deprecated":
                if l1_ratio is None:
                    emitted.append((
                        "'l1_ratio=None' was deprecated in version 1.8 and will "
                        "trigger an error in 1.10. Use 0<=l1_ratio<=1 instead.",
                        FutureWarning,
                    ))
            else:
                emitted.append((
                    "'penalty' was deprecated in version 1.8 and will be removed in "
                    "1.10. To avoid this warning, leave 'penalty' set to its default "
                    "value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of "
                    "penalty='l2', l1_ratio=1 instead of penalty='l1', "
                    + (
                        "l1_ratio set to a float between 0 and 1 instead of "
                        "penalty='elasticnet', and "
                        if _SKLEARN >= (1, 9)
                        else "and "
                    )
                    + "C=np.inf instead of penalty=None.",
                    FutureWarning,
                ))
            if resolved != "elasticnet" and l1_ratio is not None and 0 < l1_ratio < 1:
                emitted.append((
                    "l1_ratio parameter is only used when penalty is 'elasticnet'. "
                    f"Got (penalty={resolved})",
                    UserWarning,
                ))
            if (penalty == "l2" and l1_ratio != 0) or (penalty == "l1" and l1_ratio != 1):
                emitted.append((
                    f"Inconsistent values: penalty={penalty} with l1_ratio={l1_ratio}. "
                    "penalty is deprecated. Please use l1_ratio only.",
                    UserWarning,
                ))
            # 1.8 tests the resolved penalty here, 1.9 the raw parameter, so
            # LogisticRegression(C=np.inf) warns on one and not the other.
            unpenalized = penalty is None if _SKLEARN >= (1, 9) else resolved is None
            if unpenalized and C != 1.0:
                emitted.append((
                    "Setting penalty=None will ignore the C and l1_ratio parameters",
                    UserWarning,
                ))
            if self.n_jobs is not None:
                emitted.append((
                    "'n_jobs' has no effect since 1.8 and will be removed in 1.10. "
                    f"You provided 'n_jobs={self.n_jobs}', please leave it unspecified.",
                    FutureWarning,
                ))

        for message, category in emitted:
            warnings.warn(message, category, stacklevel=3 + _WRAPPER_FRAMES)

    # --------------------------------------------------------------- capability

    def _check_logistic_params(self, X) -> None:
        """The capability checks that cost nothing: parameters, and sparsity.

        Run before the crossover decision on every path, so a request mlxlearn
        cannot serve is always reported as such — with its warning and its
        ``fallback_policy`` — even when the problem is small enough that
        scikit-learn would have been used anyway.
        """
        name = type(self).__name__
        require_dense(X, estimator=name)

        params = {
            "solver": self.solver,
            "penalty": self._effective_penalty(),
            "dual": self.dual,
            "warm_start": self.warm_start,
        }
        supported = {
            "solver": _SUPPORTED_SOLVERS,
            "penalty": _SUPPORTED_PENALTIES,
            "dual": (False,),
            "warm_start": (False,),
        }

        # scikit-learn 1.8 removed multi_class outright; 1.7 still carries it as a
        # deprecated sentinel. Reading it unconditionally raised AttributeError on
        # 1.8 and 1.9 — version pairs pyproject.toml declares supported and CI
        # schedules. Checked only where it exists, rather than pinned away.
        if hasattr(self, "multi_class"):
            params["multi_class"] = self.multi_class
            supported["multi_class"] = _SUPPORTED_MULTI_CLASS

        require_supported_params(params, supported, estimator=name)

        # l1_ratio is already accounted for: on scikit-learn 1.8+ it *is* how the
        # penalty is spelled, so _effective_penalty has resolved it and anything
        # other than pure L2 was rejected above. On 1.7 the penalty is explicit
        # and scikit-learn itself ignores l1_ratio unless penalty='elasticnet',
        # so mlxlearn ignores it in exactly the same cases — being stricter here
        # would send a request to scikit-learn that mlxlearn can serve identically.
        #
        # What must not happen is the ancestor's behavior: accepting an
        # elastic-net request and quietly fitting a ridge. That cannot occur,
        # because 'elasticnet' never resolves to a supported penalty.

        # intercept_scaling only affects the liblinear solver, where it scales a
        # synthetic feature. Accepting a non-default value under lbfgs would mean
        # ignoring it.
        if self.fit_intercept and self.intercept_scaling != 1.0:
            raise UnsupportedParameterError(
                f"{name} implements intercept_scaling=1.0 only; got "
                f"{self.intercept_scaling!r}. It has an effect solely for "
                "solver='liblinear', which is not implemented here.",
                parameter="intercept_scaling",
                value=self.intercept_scaling,
            )

    def _check_logistic_capability(self, X, y, sample_weight=None) -> None:
        """Raise :class:`CapabilityError` for anything the MLX path cannot serve.

        Everything :meth:`_check_logistic_params` checks, plus the checks that have
        to look at the data. All of them run before :meth:`BackendMixin._select_backend`
        makes its decision, because only a capability error raised *inside* that
        decision becomes a Layer 2 fallback; one raised afterwards escapes as an
        exception from a class standing in for scikit-learn's.
        """
        name = type(self).__name__
        self._check_logistic_params(X)
        # Before validate_data, so a float64 value that overflows float32 is a
        # capability limit patched mode can fall back on rather than scikit-learn's
        # generic post-conversion "contains infinity" ValueError.
        check_float32_range(X)

        y_array = np.asarray(y)
        if y_array.ndim > 1 and y_array.shape[1] > 1:
            raise UnsupportedInputError(
                f"{name} implements single-output classification only; got y with "
                f"shape {y_array.shape}. Wrap the estimator in "
                "OneVsRestClassifier for multi-label targets.",
                reason="multi-label-target",
            )

        # Anything that does not read as numbers is left for validation to reject
        # with scikit-learn's own message.
        require_nonnegative_weights(sample_weight, estimator=name)

    def _below_crossover(
        self, n_samples: int, n_features: int, n_targets: int = 1
    ) -> tuple[bool, str]:
        """Whether stock scikit-learn is expected to win at this size.

        Both conditions must be met, and the work proxy is
        ``n_samples × n_features × n_targets`` — one weight vector for a binary
        problem, one per class for a multinomial one — because each objective
        evaluation is a pair of ``(n, d) × (d, n_targets)`` products. Counting only
        ``n × d`` sent a five-class 60 000 × 128 problem to scikit-learn as
        "small".

        The per-iteration cost is what the threshold has to beat. It used to be
        dominated by blocking host synchronizations inside the L-BFGS bookkeeping
        — every inner product was a device round trip of ~120 µs, 12 to 33 per
        iteration, while the line search averaged only about 1.1 evaluations per
        iteration — and that bookkeeping is host NumPy now, with one
        synchronization per objective evaluation. The thresholds themselves are
        measurements; see ``docs/benchmarks.md``.
        """
        tuning = get_tuning()
        work = n_samples * n_features * max(int(n_targets), 1)
        if n_samples < tuning.logreg_min_samples or work < tuning.logreg_min_work:
            return True, (
                f"{n_samples} samples x {n_features} features x {n_targets} "
                f"target(s) is below the measured logistic-regression crossover "
                f"({tuning.logreg_min_samples} samples and {tuning.logreg_min_work} "
                "element-ops, both required)"
            )
        return False, ""

    def _crossover_from_raw(self, shape: tuple[int, int], y) -> tuple[bool, str]:
        """:meth:`_below_crossover` from the raw input, before any validation.

        The class count only changes the answer when there are enough samples and
        ``n × d`` alone falls short of the work threshold, so ``y`` is only looked
        at in that band.
        """
        n_samples, n_features = shape
        tuning = get_tuning()
        n_targets = 1
        if (
            n_samples >= tuning.logreg_min_samples
            and n_samples * n_features < tuning.logreg_min_work
        ):
            n_targets = _count_targets(y)
        return self._below_crossover(n_samples, n_features, n_targets)

    def _select_fit_backend(self, X, y, sample_weight) -> tuple[str, bool]:
        """Decide the backend for ``fit`` once. Returns ``(backend, decided)``.

        For a patched estimator on a problem below the crossover, the decision is
        made from the raw shape after only the parameter checks: the problem is
        going to scikit-learn regardless, and scanning ``X`` for float32 overflow,
        copying it to float32 and validating ``y`` first — all of which
        scikit-learn then repeats — made the handed-off fit up to 2x slower than
        stock. Data-dependent capability limits are moot for a fit MLX will not
        run, so they are checked only when it will.

        ``decided`` is False when the input has no 2-D shape to decide on (a
        list, say); the crossover is then applied after validation.
        """
        shape = _safe_shape(X)
        if shape is None:
            backend = self._select_backend(
                "fit",
                capability_check=lambda: self._check_logistic_capability(X, y, sample_weight),
                shape=None,
            )
            return backend, False

        below, detail = self._crossover_from_raw(shape, y)
        handed_off = below and self._mlxlearn_allow_fallback and not get_tuning().force_backend
        if handed_off:
            check = lambda: self._check_logistic_params(X)  # noqa: E731
        else:
            check = lambda: self._check_logistic_capability(X, y, sample_weight)  # noqa: E731
        backend = self._select_backend(
            "fit",
            capability_check=check,
            below_crossover=below,
            crossover_detail=detail,
            shape=shape,
        )
        return backend, True

    # ---------------------------------------------------------------- inference

    @contextmanager
    def _inference(self, operation: str, X):
        """Record one diagnostics event per public inference call, then run it.

        Inference for this estimator is always scikit-learn's float64 NumPy code on
        ``coef_``, ``intercept_`` and ``classes_`` — every backend populates exactly
        those — so the event records the backend that *owns* the fitted state, with
        ``compute="numpy-float64"`` saying what actually did the arithmetic.

        A model whose attributes were assigned rather than fitted — federated
        averaging, a model imported from elsewhere — has no owner. scikit-learn
        predicts from it, so mlxlearn does too, recorded as ``"external-state"``.
        That is distinguished from a fit that started and did not complete, which
        can leave new ``classes_`` next to the previous ``coef_`` and still raises.
        """
        check_is_fitted(self)
        active = _INFERENCE_IN_PROGRESS.get()
        if id(self) not in active:
            self._record_inference(operation, _safe_shape(X))
        token = _INFERENCE_IN_PROGRESS.set(active | {id(self)})
        try:
            yield
        finally:
            _INFERENCE_IN_PROGRESS.reset(token)

    def _record_inference(self, operation: str, shape) -> None:
        backend = getattr(self, "_execution_backend_", None)
        if backend is None and not getattr(self, "_mlxlearn_fit_started", False):
            self._backend_event(
                operation,
                "sklearn",
                reason="external-state",
                detail=(
                    f"{type(self).__name__} was not fitted by mlxlearn; predicting "
                    "from its assigned coef_, intercept_ and classes_."
                ),
                shape=shape,
                compute="numpy-float64",
            )
            return
        backend = self._fitted_backend(operation)
        # Why the model was fitted where it was, as BackendMixin._note_inference
        # records it for every other estimator. The reason stays "sticky" even for
        # an MLX-fitted model on a GPU-less host: this inference is host NumPy
        # float64 on every backend, so no GPU was wanted and none was missed.
        extra = {}
        fit_event = self.__dict__.get("_fit_event_")
        if fit_event is not None:
            extra["fit_reason"] = fit_event.reason
        self._backend_event(
            operation,
            backend,
            reason="sticky",
            shape=shape,
            fallback=backend == "sklearn",
            compute="numpy-float64",
            **extra,
        )


class LogisticRegression(MLXLogisticMixin, _SkLogisticRegression):
    """L2-penalized logistic regression fitted with L-BFGS on MLX.

    A drop-in replacement for :class:`sklearn.linear_model.LogisticRegression`
    restricted to ``solver="lbfgs"`` with an L2 penalty or none, on dense,
    single-output data. The objective minimized is scikit-learn's, term for term,
    so the two agree on the objective value at their respective solutions.

    Parameters
    ----------
    penalty : {"l2", None}, default="l2" on scikit-learn 1.7, "deprecated" on 1.8+
        On 1.8+ this is a deprecated sentinel and the penalty is spelled with
        ``l1_ratio`` and ``C`` instead — ``l1_ratio=0`` for L2, ``C=np.inf`` for
        none — exactly as in scikit-learn. ``"l1"`` and ``"elasticnet"``, however
        spelled, are not implemented and raise.
    dual : bool, default=False
        Only ``False`` is implemented.
    tol : float, default=1e-4
        Tolerance on ``max|gradient|``, the same criterion and scale scikit-learn's
        lbfgs path uses. A value below what float32 can resolve — ``0``, say — is
        honored as "run to the float32 floor"; see Notes.
    C : float, default=1.0
        Inverse regularization strength. ``np.inf`` means unpenalized.
    fit_intercept : bool, default=True
    intercept_scaling : float, default=1
        Only ``1.0`` is accepted with an intercept; it affects nothing outside
        ``solver="liblinear"``.
    class_weight : dict or "balanced", default=None
        Folded into the sample weights exactly as scikit-learn folds them, before
        the penalty scale is derived from their sum.
    random_state : int, RandomState or None, default=None
        Accepted and unused: L-BFGS from a zero start is deterministic.
    solver : {"lbfgs"}, default="lbfgs"
    max_iter : int, default=100
    multi_class : str, default="deprecated"
        scikit-learn 1.7 only; 1.8 removed it. Only the default is implemented —
        multinomial for three or more classes, a single weight vector for two — and
        ``"auto"``, which means the same. ``"ovr"`` and ``"multinomial"`` raise
        rather than being reinterpreted.
    verbose : int, default=0
        Accepted and ignored.
    warm_start : bool, default=False
        Only ``False`` is implemented.
    n_jobs : int, default=None
        Accepted and ignored; the MLX path is not joblib-parallel. scikit-learn
        1.8+ deprecates it and warns when it is set, and so does this class.
    l1_ratio : float, default=None on scikit-learn 1.7, 0.0 on 1.8+
        On 1.8+ it selects the penalty, and only ``0`` (or the deprecated
        ``None``) is implemented. On 1.7 it is ignored unless
        ``penalty="elasticnet"``, as scikit-learn ignores it.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Ascending, with the dtype of ``y``.
    coef_ : ndarray of shape (1, n_features) or (n_classes, n_features)
        float64. Binary problems get a single weight vector for ``classes_[1]``,
        matching scikit-learn.
    intercept_ : ndarray of shape (1,) or (n_classes,)
        float64. Zeros when ``fit_intercept=False``.
    n_iter_ : ndarray of shape (1,), dtype int32
        Accepted L-BFGS steps. Always length one, because there is no
        one-versus-rest path that would fit several independent problems.
    n_features_in_ : int
    execution_backend_ : {"mlx", "cpu", "sklearn"}

    Notes
    -----
    **Precision.** Computation is float32 (``docs/fp32_policy.md``); ``coef_`` and
    ``intercept_`` are float64 containers. Float64 input is centered in float64
    before it is rounded, so a large constant offset — timestamps, coordinates —
    costs no more precision than the spread of the data does. The objective is
    evaluated through ``logaddexp``/``logsumexp`` and never exponentiates a raw
    score, so separable data with logits in the hundreds produces finite losses
    rather than the floored plateaus that a ``clip``-based formulation gives in
    float32.

    **Convergence is reported.** A run that exhausts ``max_iter`` or stalls emits
    :class:`sklearn.exceptions.ConvergenceWarning`, as scikit-learn does, and
    records the stop on the fit's diagnostics event: ``fit_event_.extra`` gains
    ``stop_reason`` and the solver's state. The exception is a run that stops
    because the objective can no longer resolve progress with ``max|gradient|``
    already below about 1e-6 of its starting value: that is the float32 floor
    ``docs/fp32_policy.md`` §2.3 describes, reached rather than missed, so it is
    recorded without a warning. ``n_iter_`` counts accepted steps, so a problem
    already optimal at the zero start reports ``0``.

    **Callbacks.** On scikit-learn 1.9+, callbacks set with ``set_callbacks`` see
    the same task tree as scikit-learn's lbfgs path: the fit, and one subtask per
    accepted L-BFGS iteration carrying that iterate's ``coef_`` and
    ``intercept_``. Like scikit-learn 1.9, a request to stop early is not acted on.

    **No warm start, no caches.** ``fit`` is a pure function of ``(X, y,
    sample_weight, params)``. Nothing about a previous fit in the same process can
    reach this one.

    Examples
    --------
    >>> import numpy as np
    >>> from mlxlearn.linear_model import LogisticRegression
    >>> X = np.array([[0.0], [1.0], [2.0], [3.0]])
    >>> y = np.array([0, 0, 1, 1])
    >>> clf = LogisticRegression().fit(X, y)
    >>> clf.coef_.shape
    (1, 1)
    >>> clf.predict([[2.5]])
    array([1])
    """

    @_with_callbacks
    def fit(self, X, y, sample_weight=None):
        """Fit the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        sample_weight : array-like of shape (n_samples,), default=None

        Returns
        -------
        self : LogisticRegression
        """
        # scikit-learn applies its parameter constraints through a private fit
        # decorator this class does not use, so they are invoked explicitly. The
        # ancestor skipped them entirely, which turned C=-1 into a silent clamp to
        # 1e-12 instead of the InvalidParameterError scikit-learn raises.
        self._validate_params()

        self._clear_backend()
        # Distinguishes "a fit started here" from "coef_ was assigned from outside"
        # for inference; see MLXLogisticMixin._inference. Deliberately not a
        # trailing-underscore name, which check_is_fitted would read as fitted state.
        self._mlxlearn_fit_started = True

        backend, decided = self._select_fit_backend(X, y, sample_weight)
        if backend == "sklearn":
            return self._fit_with_sklearn(X, y, sample_weight)

        # float64 is kept as float64 — the solver centers it before rounding to
        # float32, see _solvers/logistic.py — and float32 input is not upcast.
        X_checked, y_checked = validate_data(
            self,
            X,
            y,
            accept_sparse=False,
            dtype=[np.float64, np.float32],
            order="C",
            ensure_min_samples=1,
            reset=True,
        )
        _check_classification_target(y_checked)

        self.classes_, y_encoded = np.unique(y_checked, return_inverse=True)
        n_classes = self.classes_.shape[0]
        if n_classes < 2:
            # scikit-learn's wording, verbatim: code written against its error
            # contract matches on this string.
            raise ValueError(
                "This solver needs samples of at least 2 classes in the data, but "
                f"the data contains only one class: {self.classes_[0]!r}"
            )

        if not decided:
            below, detail = self._below_crossover(
                *X_checked.shape, n_targets=1 if n_classes == 2 else n_classes
            )
            # No backend != "cpu" exemption: device never overrides the
            # crossover, and _select_backend lets MLXLEARN_FORCE_BACKEND outrank it.
            if below:
                backend = self._select_backend(
                    "fit",
                    below_crossover=True,
                    crossover_detail=detail,
                    shape=X_checked.shape,
                )
                if backend == "sklearn":
                    return self._fit_with_sklearn(X, y, sample_weight)

        self._mirror_fit_warnings()
        weights = self._resolve_weights(X_checked, y_checked, y_encoded, sample_weight)

        # Resolved rather than read, so that C=inf on scikit-learn 1.8+ means
        # "unpenalized" here exactly as it does there.
        C = math.inf if self._effective_penalty() is None else float(self.C)

        # "mlx" is the MLX GPU backend. A fit forced onto it in a process with no
        # GPU runs the same float32 code on MLX's CPU, as the other estimators'
        # kernels do (inference_device) — unless device="gpu" was named, which
        # use_device then refuses.
        device = "cpu"
        if backend == "mlx":
            device = "gpu" if get_config().device == "gpu" else inference_device(backend)

        tuning = get_tuning()
        config = LogisticConfig(
            C=C,
            fit_intercept=bool(self.fit_intercept),
            tol=float(self.tol),
            max_iter=int(self.max_iter),
            history_size=tuning.lbfgs_history,
            max_line_search=tuning.lbfgs_max_line_search,
            device=device,
        )

        callback_ctx = bridge = None
        metadata = None if sample_weight is None else {"sample_weight": sample_weight}
        if hasattr(self, "_init_callback_context"):  # scikit-learn >= 1.9
            # The same task tree scikit-learn's lbfgs path builds: the fit, one
            # subtask per iteration, plus the trailing empty one its scipy bridge
            # always leaves.
            callback_ctx = self._init_callback_context(max_subtasks=max(self.max_iter, 1) + 1)
            callback_ctx.call_on_fit_task_begin(
                estimator=self, X=X_checked, y=y_checked, metadata=metadata
            )
            if hasattr(self, "_skl_callbacks"):
                bridge = _CallbackBridge(
                    callback_ctx, self, X_checked, y_checked, metadata, backend,
                    n_targets=1 if n_classes == 2 else n_classes,
                    n_features=X_checked.shape[1],
                )

        callback_error = None
        try:
            with mlx_guard(type(self).__name__, "fit"):
                solution = solve_logistic(
                    X_checked,
                    y_encoded.astype(np.int32, copy=False),
                    n_classes=n_classes,
                    sample_weight=weights,
                    config=config,
                    callback=bridge,
                )
        except BackendExecutionError as exc:
            if bridge is None or bridge.error is None or exc.original is not bridge.error:
                raise
            callback_error = bridge.take_error()
        if callback_error is not None:
            # The user's callback raised, not MLX: re-raise exactly that. Outside
            # the handler, which would replace its own __cause__ and __context__
            # with the BackendExecutionError, and without keeping it in this frame,
            # which its traceback holds.
            try:
                raise callback_error
            finally:
                callback_error = None

        if bridge is not None:
            bridge.close()

        self.coef_ = solution.coef
        self.intercept_ = solution.intercept
        self.n_iter_ = np.array([solution.n_iter], dtype=np.int32)
        # Before the commit, so a ConvergenceWarning that a user's "error" filter
        # escalates leaves the fit incomplete, as every fit that raises does.
        self._report_convergence(solution)
        self._set_backend(backend)

        if callback_ctx is not None:
            callback_ctx.call_on_fit_task_end(
                estimator=self,
                X=X_checked,
                y=y_checked,
                metadata=metadata,
                reconstruction_attributes={},
            )
        return self

    def _fit_with_sklearn(self, X, y, sample_weight):
        """Hand the whole fit to scikit-learn, which manages its own fitted state.

        The in-progress markers go first: scikit-learn 1.9 builds mid-fit copies
        of this estimator (``copy.copy``) for callbacks, and those have to be able
        to predict, so while scikit-learn fits they are judged by its own
        fitted-attribute rule. A scikit-learn fit that fails partway reports the
        estimator unfitted again, as every other failed refit does
        (:meth:`BackendMixin.__sklearn_is_fitted__`), rather than predicting from
        new ``classes_`` beside the previous ``coef_``.
        """
        state = self.__dict__
        state.pop("_mlxlearn_fit_started", None)
        state.pop("_mlxlearn_fit_incomplete", None)
        try:
            super().fit(X, y, sample_weight)
        except BaseException:
            self._begin_fit()
            raise
        self._set_backend("sklearn")
        return self

    def _report_convergence(self, solution) -> None:
        """Warn about a run that did not converge, and record every such stop.

        The ancestor unpacked ``converged`` and discarded it at every call site, so
        a model that stopped at max_iter was indistinguishable from one that
        reached the tolerance.

        The stop is recorded on the fit's own diagnostics event, which
        :meth:`_set_backend` commits next: its ``detail`` gains the explanation and
        its ``extra`` the ``stop_reason``, ``n_iter``, ``n_fev``, ``grad_max_abs``,
        ``tol`` and ``at_resolution_floor``. As an event of its own it broke "one
        fit, one event": it became ``get_last_backend_event()`` in place of the
        fit's, and counted as a second piece of work in ``diagnostics_summary()``.
        """
        if solution.converged:
            return
        floor = solution.stop_reason == "float32-floor" and solution.at_resolution_floor
        state = (
            f"after {solution.n_iter} iteration(s) ({solution.stop_reason}): max|gradient| "
            f"is {solution.grad_max_abs:.3g} against tol={self.tol:g}"
        )
        # Every warning starts the way scikit-learn's lbfgs warning does, so a filter
        # or a pytest.warns(match=...) written against "lbfgs failed to converge"
        # still matches when patch_sklearn() sends the fit to MLX.
        if floor:
            detail = (
                f"lbfgs ran to the float32 floor {state}, below the ~1e-6 x "
                f"{solution.initial_grad_max_abs:.3g} that float32 resolves "
                "(docs/fp32_policy.md §2.3)."
            )
        elif solution.stop_reason == "max_iter":
            detail = (
                f"lbfgs failed to converge {state}. Increase max_iter, scale the data, "
                "or relax tol."
            )
        elif solution.stop_reason == "float32-floor":
            detail = (
                f"lbfgs failed to converge {state}; the objective stopped resolving "
                "progress in float32 before the gradient reached tol. Scale the data, "
                "or relax tol."
            )
        elif solution.stop_reason == "line-search-failure":
            detail = (
                f"lbfgs failed to converge {state}; no step along the search direction "
                "decreased the objective. Scale the data, or relax tol."
            )
        else:
            detail = (
                f"lbfgs failed to converge {state}. The objective is not finite at the "
                "starting point; check the data for extreme values, or scale it."
            )

        # The decision _select_backend staged, on every path that reaches the solver.
        pending = self.__dict__["_mlxlearn_pending_event"]
        pending["detail"] = f"{pending.get('detail', '')} {detail}".lstrip()
        pending.update(
            stop_reason=solution.stop_reason,
            n_iter=solution.n_iter,
            n_fev=solution.n_fev,
            grad_max_abs=solution.grad_max_abs,
            tol=float(self.tol),
            at_resolution_floor=solution.at_resolution_floor,
        )
        if not floor:
            warnings.warn(detail, ConvergenceWarning, stacklevel=3 + _WRAPPER_FRAMES)

    # ------------------------------------------------------------------ weights

    def _resolve_weights(self, X, y, y_encoded, sample_weight):
        """Combine ``sample_weight`` and ``class_weight`` the way scikit-learn does.

        The order matters and is not arbitrary: class weights multiply the sample
        weights *first*, and only then is ``sw_sum`` taken, because the penalty
        scale is ``1/(C·sw_sum)``. Folding them in the other order changes the
        effective regularization, and scikit-learn has a test asserting that
        passing ``class_weight`` is equivalent to passing the corresponding
        ``sample_weight``.
        """
        weights = check_sample_weight(sample_weight, X, dtype=np.float64)
        if self.class_weight is None:
            return weights

        if weights is None:
            weights = np.ones(X.shape[0], dtype=np.float64)
        else:
            weights = weights.copy()
        # compute_class_weight is given the sample weights as well, because
        # "balanced" means "inversely proportional to the *weighted* class
        # frequencies" in scikit-learn 1.7.
        class_weights = compute_class_weight(
            self.class_weight, classes=self.classes_, y=y, sample_weight=weights
        )
        weights *= np.asarray(class_weights, dtype=np.float64)[y_encoded]
        return weights

    # ---------------------------------------------------------------- inference

    # scikit-learn computes all of these from coef_, intercept_ and classes_, in
    # float64, and an MLX fit populates exactly those attributes with exactly those
    # shapes. Reimplementing them in MLX would trade float64 exactness for nothing:
    # the work is one (n, d) x (d, K) matmul, far below any crossover. So these
    # overrides add only the sticky-backend bookkeeping — one diagnostics event per
    # public call, see MLXLogisticMixin._inference — and then defer. The ancestor
    # did reimplement them, in float32, with `score >= 0` where scikit-learn uses
    # `score > 0`, so its predict and its decision_function could disagree on the
    # decision boundary.

    def decision_function(self, X):
        """Confidence scores, ``X @ coef_.T + intercept_``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        scores : ndarray of shape (n_samples,) or (n_samples, n_classes)
            One-dimensional for binary problems.
        """
        with self._inference("decision_function", X):
            return super().decision_function(X)

    def predict(self, X):
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y : ndarray of shape (n_samples,)
            Labels drawn from ``classes_``.
        """
        with self._inference("predict", X):
            return super().predict(X)

    def predict_proba(self, X):
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            float64, rows sum to 1, columns ordered by ``classes_``.
        """
        with self._inference("predict_proba", X):
            return super().predict_proba(X)

    def predict_log_proba(self, X):
        """Predict the logarithm of class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        log_p : ndarray of shape (n_samples, n_classes)
        """
        with self._inference("predict_log_proba", X):
            return super().predict_log_proba(X)

    def score(self, X, y, sample_weight=None):
        """Mean accuracy on the given data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        sample_weight : array-like of shape (n_samples,), default=None

        Returns
        -------
        score : float
        """
        with self._inference("score", X):
            return super().score(X, y, sample_weight=sample_weight)


class _CallbackBridge:
    """Drives scikit-learn 1.9 callbacks from the L-BFGS iterations.

    The same hook sequence as scikit-learn's ``_LbfgsCallbackBridge``: a subtask is
    begun with the starting point, each accepted step ends the current subtask
    and begins the next, and :meth:`close` ends the last one as an empty task.
    Each iterate is handed over with the attributes a callback needs to build a
    usable estimator from it — including the backend that owns it, so that the
    copy scikit-learn builds mid-fit can predict.

    The per-iteration hooks run inside the solver, and so inside ``mlx_guard``,
    which reports anything but mlxlearn's own errors as an MLX failure. The bridge
    therefore keeps what a hook raised (:attr:`error`), and ``fit`` unwraps the
    ``BackendExecutionError`` built around exactly that and re-raises it unchanged:
    a callback's error is the caller's, not a backend bug.
    """

    def __init__(self, ctx, estimator, X, y, metadata, backend, *, n_targets, n_features):
        self._ctx = ctx
        self._estimator = estimator
        self._X = X
        self._y = y
        self._metadata = metadata
        self._backend = backend
        #: What the last per-iteration hook raised, if it did not return, and the
        #: traceback it had as it left the bridge.
        self.error: BaseException | None = None
        self._traceback = None
        self._iteration = self._begin(
            np.zeros((n_targets, n_features)), np.zeros(n_targets)
        )

    def _attributes(self, coef, intercept):
        return {"coef_": coef, "intercept_": intercept, "_execution_backend_": self._backend}

    def _begin(self, coef, intercept):
        context = self._ctx.subcontext(task_name="lbfgs-iter")
        context.call_on_fit_task_begin(
            estimator=self._estimator,
            X=self._X,
            y=self._y,
            metadata=self._metadata,
            reconstruction_attributes=self._attributes(coef, intercept),
        )
        return context

    def __call__(self, coef, intercept):
        returned = False
        try:
            self._iteration.call_on_fit_task_end(
                estimator=self._estimator,
                X=self._X,
                y=self._y,
                metadata=self._metadata,
                reconstruction_attributes=self._attributes(coef, intercept),
            )
            self._iteration = self._begin(coef, intercept)
            returned = True
        finally:
            # Only noted: whatever was raised propagates untouched from here.
            self.error, self._traceback = (None, None) if returned else sys.exc_info()[1:]

    def take_error(self) -> BaseException:
        """Hand over what a hook raised, with the traceback it left the bridge with.

        Past the bridge it also picked up ``mlx_guard``'s generator frame, whose
        caller's frame holds the exception itself: a reference cycle that kept the
        failed fit's frames, device copy of ``X`` included, alive until the garbage
        collector next ran, rather than until the caller let go of the exception.
        """
        error, traceback = self.error, self._traceback
        self.error = self._traceback = None
        return error.with_traceback(traceback)

    def close(self):
        self._iteration.call_on_fit_task_end(estimator=self._estimator)


def _count_targets(y) -> int:
    """Weight vectors the fit will have: 1 for binary, else the class count.

    Used only as a size hint for the crossover, before validation, so anything it
    cannot read cheaply counts as binary — the smaller work estimate, which errs
    toward handing the problem to scikit-learn.
    """
    labels = np.asarray(y)
    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels.ravel()
    if labels.ndim != 1 or labels.dtype.kind not in "biufUS":
        return 1
    n_classes = np.unique(labels).shape[0]
    return 1 if n_classes <= 2 else int(n_classes)


def _check_classification_target(y: np.ndarray) -> None:
    """Reject non-classification targets with scikit-learn's own message and warning.

    Multi-output targets are a capability limit, not an invalid request, so they
    get :class:`~mlxlearn.exceptions.UnsupportedInputError`; everything else goes
    through ``check_classification_targets``, which raises scikit-learn's exact
    "Unknown label type" error and emits its "more than 50% unique classes"
    warning.
    """
    target_type = type_of_target(y, input_name="y")
    if target_type in ("multilabel-indicator", "multiclass-multioutput", "continuous-multioutput"):
        raise UnsupportedInputError(
            "LogisticRegression implements single-output classification only; got a "
            f"{target_type!r} target.",
            reason="multi-label-target",
        )
    check_classification_targets(y)


def _safe_shape(X) -> tuple[int, ...] | None:
    shape = getattr(X, "shape", None)
    if shape is None or len(shape) != 2:
        return None
    return (int(shape[0]), int(shape[1]))
