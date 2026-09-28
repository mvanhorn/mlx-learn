"""The penalized logistic objective and its gradient, in MLX.

Same contract as every solver here: arrays and a frozen :class:`LogisticConfig`
in, a frozen :class:`LogisticSolution` out. No estimator, no scikit-learn import,
no global configuration.

The objective is scikit-learn's, exactly
------------------------------------------
For weights ``W`` (the intercept excluded) and per-sample weights ``sw`` summing to
``sw_sum``::

    J(W, b) = (1 / sw_sum) * Σᵢ swᵢ · lossᵢ  +  0.5 · (1 / (C · sw_sum)) · ‖W‖²

That is scikit-learn's ``LinearModelLoss`` with ``l2_reg_strength = 1/(C·sw_sum)``,
where ``sw_sum`` is ``n_samples`` when unweighted and the sum of the sample weights
*after* class weights have been folded in otherwise. The intercept is never
penalized. Matching this expression exactly is what makes objective-value parity a
meaningful assertion: two solvers minimizing the same function must agree on the
value at their respective solutions even when they disagree on the argument.

``lossᵢ`` is the negative log-likelihood — ``HalfBinomialLoss`` and
``HalfMultinomialLoss`` in scikit-learn's vocabulary::

    binary       log(1 + exp(raw)) − y·raw          y ∈ {0, 1}
    multinomial  logsumexp(raw) − raw[y]

Numerical stability is not optional here
----------------------------------------
Both forms above are written with ``logaddexp``/``logsumexp`` and never exponentiate
a raw score. The ancestor computed ``sigmoid(clip(score, -80, 80))`` and then
``log(clip(p, 1e-12, 1))``; in float32 a sigmoid saturates to exactly 1.0 by
``raw ≈ 17``, so every confidently classified sample contributed the same floored
``−log(1e-12) = 27.63`` and the loss surface acquired flat plateaus that no line
search can descend. Worse, its *gradient* was the exact derivative of the
*unclipped* objective, so the value and the slope described different functions —
which is what produced line-search failures reported as ordinary non-convergence.

Here the value and the gradient are derived from the same expression, and
``p − y`` is obtained from ``mx.sigmoid`` / ``exp(raw − logsumexp(raw))``, both of
which are the exact derivatives of the stable loss forms.

Layout
------
The parameter is a single ``(n_targets, n_features + fit_intercept)`` MLX array;
:func:`~mlxlearn._solvers.lbfgs.minimize` is shape-agnostic, so there is no ravel
and no chance of the feature-major/class-major transposition hazard that made the
ancestor's warm-start vectors silently meaningless. ``n_targets`` is 1 for binary
— scikit-learn fits a single weight vector for the positive class ``classes_[1]``,
not a two-row softmax — and ``n_classes`` otherwise.

On the GPU the binary design is stored transposed. MLX's vector-matrix product
``(n,) @ (n, d)`` — the binary gradient ``residualᵀX`` — takes a path that costs
up to 38x the forward ``X @ w`` on tall, narrow data, while a matrix-vector product
against a contiguous ``Xᵀ`` costs the same as the forward pass. So the design is
transposed once on the device, the untransposed copy is dropped, and both products
run as ``Xᵀ``-matvecs. On the CPU device the transpose itself costs more than it
saves, and the multinomial gradient is already a matrix-matrix product, so neither
changes.

Exact rescalings
----------------
Three transformations are applied before any float32 arithmetic, all of which are
exact — they change how the problem is represented, never which problem is solved:

* **Centering, before the rounding to float32.** See :func:`_prepare_design`.
* **Sample weights divided by a power of two**, so the largest is in ``[1, 2)``.
  Weights of 1e36 are legal and scikit-learn fits them, but the float32 reductions
  of ``Σ swᵢ·xᵢ·rᵢ`` overflow long before the weights themselves do — from about
  1e36 at 2000 rows. The data term is normalized by the equally scaled
  ``sw_sum``, so ``(1/sw_sum)·Σ sw·loss`` is unchanged, and the penalty is computed
  from the *unscaled* sum, so ``l2`` is unchanged too.
* **A diagonal preconditioner from the column scales**, handed to L-BFGS as its
  initial inverse Hessian. It rescales nothing in the data; it tells the solver that
  a feature measured in units of 1e8 has curvature 1e16 times the intercept's.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .._common.arrays import NUMPY_FLOAT
from .._common.device import use_device
from .lbfgs import GRADIENT_RESOLUTION, LBFGSConfig, minimize

__all__ = [
    "LogisticConfig",
    "LogisticSolution",
    "solve_logistic",
    "reference_objective",
]


@dataclass(frozen=True, slots=True)
class LogisticConfig:
    """Problem-independent settings for one logistic fit.

    Parameters
    ----------
    C : float, default=1.0
        Inverse regularization strength, scikit-learn's meaning. ``math.inf``
        expresses an unpenalized fit, which is how scikit-learn itself encodes
        ``penalty=None``.
    fit_intercept : bool, default=True
        The intercept column is appended to the parameter array and excluded from
        the penalty.
    tol : float, default=1e-4
        Forwarded to L-BFGS as the tolerance on ``max|gradient|``.
    max_iter : int, default=100
    history_size : int, default=10
    max_line_search : int, default=32
    device : {"auto", "gpu", "cpu"}, default="auto"
        Passed straight to :func:`~mlxlearn._common.device.use_device`. Explicit,
        because a solver that reads global configuration cannot be tested in
        isolation.
    """

    C: float = 1.0
    fit_intercept: bool = True
    tol: float = 1e-4
    max_iter: int = 100
    history_size: int = 10
    max_line_search: int = 32
    device: str = "auto"

    def __post_init__(self) -> None:
        if not (self.C > 0.0):
            raise ValueError(f"C must be strictly positive, got {self.C!r}")
        if self.max_iter < 0:
            raise ValueError(f"max_iter must be non-negative, got {self.max_iter}")
        if self.tol < 0 or not math.isfinite(self.tol):
            raise ValueError(f"tol must be finite and non-negative, got {self.tol}")


@dataclass(frozen=True, slots=True)
class LogisticSolution:
    """Fitted coefficients and everything needed to explain the run.

    Attributes
    ----------
    coef : ndarray of shape (n_targets, n_features), float64
    intercept : ndarray of shape (n_targets,), float64
        Zeros when ``fit_intercept=False``, matching scikit-learn.
    n_iter : int
        Accepted L-BFGS steps.
    objective : float
        Penalized objective at the solution, as computed in float32.
    grad_max_abs : float
        ``max|gradient|`` at the solution — the quantity ``tol`` bounds.
    converged : bool
    stop_reason : str
        See :class:`~mlxlearn._solvers.lbfgs.LBFGSState`.
    n_fev : int
        Objective/gradient evaluations, including the one at the start.
    initial_grad_max_abs : float
        ``max|gradient|`` at the zero start, the scale the float32 gradient floor is
        relative to.
    """

    coef: np.ndarray
    intercept: np.ndarray
    n_iter: int
    objective: float
    grad_max_abs: float
    converged: bool
    stop_reason: str
    n_fev: int = 0
    initial_grad_max_abs: float = math.nan

    @property
    def at_resolution_floor(self) -> bool:
        """Whether ``max|g|`` is below what float32 can resolve (fp32_policy §2.3)."""
        return self.grad_max_abs <= GRADIENT_RESOLUTION * self.initial_grad_max_abs


def _l2_reg_strength(C: float, sw_sum: float) -> float:
    """``1/(C·sw_sum)``, with ``C = inf`` meaning "no penalty".

    scikit-learn's own ``penalty=None`` path sets ``C_ = np.inf`` and reuses the
    same expression, so this reproduces it rather than branching on a separate
    flag that could disagree with it.
    """
    if math.isinf(C):
        return 0.0
    return 1.0 / (C * sw_sum)


# --------------------------------------------------------------------- objectives

#: Partial sums brought back per objective evaluation instead of one total.
#:
#: The line search compares objective values, so the *resolution* of the returned
#: value sets how far the solver can descend before every candidate step looks
#: like a tie. A float32 total of ``n`` log-loss terms has an absolute error of
#: order ``eps·Σloss``; divided by ``sw_sum`` that leaves roughly ``5e-8`` of
#: resolution on a mean loss of ``0.54``, and the last few L-BFGS iterations
#: — the ones that take ``max|g|`` from ``2e-4`` down past ``1e-4`` — move the
#: objective by about ``1e-8``. Measured: the solver stalled at ``max|g| = 1.9e-4``
#: after 20 iterations and then spent 80 more taking null steps.
#:
#: Reducing to this many partial sums on the device and finishing the addition in
#: float64 on the host cuts that error by ``√n_blocks``, to about ``1e-9``, which
#: is enough to reach scikit-learn's own ``gtol``. It costs one 4 KiB transfer per
#: evaluation — the same synchronization that reading the scalar already required.
_SUM_BLOCKS = 1024


def _blocked_sum(mx, values):
    """Reduce ``values`` to at most ``_SUM_BLOCKS`` partial sums, on the device."""
    n = values.shape[0]
    if n <= _SUM_BLOCKS:
        return values
    chunk = -(-n // _SUM_BLOCKS)
    padding = chunk * _SUM_BLOCKS - n
    if padding:
        # Zero padding is exact and cannot shift the total.
        values = mx.pad(values, (0, padding))
    return mx.sum(mx.reshape(values, (_SUM_BLOCKS, chunk)), axis=1)


def _total(parts, penalty, sw_sum: float) -> float:
    """Finish the reduction in float64 on the host. Assumes both are evaluated."""
    value = float(np.asarray(parts).astype(np.float64).sum() / sw_sum)
    return value if penalty is None else value + float(penalty)


def _binary_objective(mx, X, y, sw, sw_sum, l2, fit_intercept, n_features, *, transposed=False):
    """``fun(theta) -> (value, grad)`` for the two-class log-loss.

    ``theta`` is ``(1, n_features + fit_intercept)``. With ``transposed=True``, ``X``
    is the contiguous ``(n_features, n_samples)`` transpose and both products are
    matrix-vector products against it; see the module docstring for why.
    ``sw_sum`` is the normalizer of the data term — the sum of the weights as they
    are passed here.
    """
    zero = mx.zeros((1,), dtype=X.dtype)

    def fun(theta):
        w = theta[0, :n_features]
        raw = w @ X if transposed else X @ w
        if fit_intercept:
            raw = raw + theta[0, n_features]

        # log(1 + exp(raw)) − y·raw, evaluated without ever forming exp(raw).
        loss = mx.logaddexp(zero, raw) - y * raw
        residual = mx.sigmoid(raw) - y
        if sw is not None:
            loss = loss * sw
            residual = residual * sw

        grad_w = ((X @ residual) if transposed else (residual @ X)) / sw_sum
        if l2:
            grad_w = grad_w + l2 * w
        if fit_intercept:
            grad_b = mx.sum(residual) / sw_sum
            grad = mx.concatenate([grad_w, mx.reshape(grad_b, (1,))])
        else:
            grad = grad_w
        grad = mx.reshape(grad, (1, theta.shape[1]))

        parts = _blocked_sum(mx, loss)
        penalty = mx.sum(w * w) * (0.5 * l2) if l2 else None
        # One synchronization for the value and the gradient together.
        if penalty is None:
            mx.eval(parts, grad)
        else:
            mx.eval(parts, penalty, grad)
        return _total(parts, penalty, sw_sum), grad

    return fun


def _multinomial_objective(mx, X, y_index, y_onehot, sw, sw_sum, l2, fit_intercept, n_features):
    """``fun(theta) -> (value, grad)`` for the softmax cross-entropy.

    ``theta`` is ``(n_classes, n_features + fit_intercept)``; ``sw_sum`` as for
    :func:`_binary_objective`.
    """

    def fun(theta):
        W = theta[:, :n_features]
        raw = X @ W.T
        if fit_intercept:
            raw = raw + theta[:, n_features]

        # logsumexp(raw) − raw[y]: the shifted-softmax identity, done by MLX, so
        # the per-sample loss stays exact for logits of any magnitude instead of
        # flooring once a probability underflows.
        log_norm = mx.logsumexp(raw, axis=1)
        correct = mx.take_along_axis(raw, y_index, axis=1)[:, 0]
        loss = log_norm - correct
        residual = mx.exp(raw - mx.reshape(log_norm, (-1, 1))) - y_onehot
        if sw is not None:
            loss = loss * sw
            residual = residual * mx.reshape(sw, (-1, 1))

        grad_W = (residual.T @ X) / sw_sum
        if l2:
            grad_W = grad_W + l2 * W
        if fit_intercept:
            grad_b = mx.sum(residual, axis=0) / sw_sum
            grad = mx.concatenate([grad_W, mx.reshape(grad_b, (-1, 1))], axis=1)
        else:
            grad = grad_W

        parts = _blocked_sum(mx, loss)
        penalty = mx.sum(W * W) * (0.5 * l2) if l2 else None
        if penalty is None:
            mx.eval(parts, grad)
        else:
            mx.eval(parts, penalty, grad)
        return _total(parts, penalty, sw_sum), grad

    return fun


# ------------------------------------------------------------------------- solve


#: Rows sampled to estimate the column scales for the preconditioner. It is rounded
#: to a power of four, so an estimate is all it needs; the answer never depends on it.
_SCALE_SAMPLE_ROWS = 4096


def _prepare_design(X, fit_intercept: bool):
    """Center before rounding to float32. Returns ``(X32, center, scale)``.

    Centering with an intercept is exact, not an approximation:
        w·(x − μ) + b′  ≡  w·x + (b′ − w·μ)
    so the model is unchanged and the intercept is corrected afterwards, with the
    ``μ`` that was actually subtracted.

    Without it, features carrying a large constant offset destroy the fit in
    float32. Measured on 2000 x 8 with an offset of 1e4 added to every feature: the
    solver stopped after 3 iterations with an all-zero coefficient vector, an
    objective of exactly log 2 — the null model — and 0.50 accuracy, where
    scikit-learn reached 0.62. The gradient of the offset-dominated design is
    numerically parallel to the constant direction, so the first step goes almost
    entirely into the intercept and the line search then cannot find a decrease.

    The order matters as much as the centering. Rounding float64 input to float32
    first quantizes the *offset* — 0.0625 at 1e6, 128 for a Unix timestamp — and the
    informative part of each feature with it, before the subtraction meant to
    protect it: coefficients came out 4x attenuated on two timestamp columns. So
    float64 input is centered in float64 and only the centered values are rounded.
    NumPy buffers the mixed-precision subtraction internally, so no float64
    temporary the size of ``X`` exists. Float32 input is centered by the float32
    mean instead, in float32: for offset-dominated data the subtraction is then
    exact (Sterbenz), and the rounding of the mean itself is a constant shift per
    column that the intercept absorbs — provided the un-centering uses that same
    rounded mean, which it does.

    Centering cannot be applied when fit_intercept is False: there is no intercept
    to absorb the shift, so the shifted problem is a *different* problem, and
    silently solving it would be wrong. That case keeps the raw design.

    ``scale`` estimates each column's root-mean-square after centering, the input
    to :func:`_preconditioner`.
    """
    source = np.asarray(X)
    if source.dtype not in (np.float32, np.float64):
        source = source.astype(np.float64)
    source = np.ascontiguousarray(source)
    n_samples = source.shape[0]

    center = None
    if fit_intercept and n_samples > 0:
        center = source.mean(axis=0, dtype=np.float64)
        if source.dtype == np.float32:
            center = center.astype(np.float32).astype(np.float64)
        if not np.any(center != 0.0):
            center = None

    if center is None:
        design = source.astype(NUMPY_FLOAT, copy=False)
    elif source.dtype == np.float32:
        design = source - center.astype(np.float32)
    else:
        design = np.empty(source.shape, dtype=NUMPY_FLOAT)
        np.subtract(source, center, out=design, casting="same_kind")

    sample = design[:: max(1, n_samples // _SCALE_SAMPLE_ROWS)].astype(np.float64)
    scale = np.sqrt(np.einsum("ij,ij->j", sample, sample) / max(sample.shape[0], 1))
    return design, center, scale


#: Powers of four of curvature mismatch left to the quasi-Newton update; see
#: :func:`_preconditioner`.
_PRECONDITION_DEADBAND = 4


def _preconditioner(scale: np.ndarray, l2: float, n_targets: int, fit_intercept: bool) -> np.ndarray:
    """The diagonal initial inverse Hessian L-BFGS starts each iteration from.

    The curvature along a feature's coefficient is about ``p(1-p)·scale² + l2``,
    against ``p(1-p)`` for the intercept, so a feature in units of 1e8 beside an
    intercept of order one is a condition number of 1e16 that the quasi-Newton
    update has to discover one direction at a time. Measured on 3000 x 6 with every
    feature scaled by 1e5 or more: the coefficients converged, the intercept never
    moved, and the objective stopped 1.4e-3 above the optimum. The ratio of the two
    curvatures at the zero start, ``1 / (scale² + 4·l2)``, removes that; the
    ``l2`` term keeps a tiny, penalized column from being given a huge step the
    penalty's own curvature would not allow.

    Rounded to a power of four and shrunk toward the identity by
    ``_PRECONDITION_DEADBAND`` of them: a column within a factor of 4⁴ of the
    intercept's curvature gets exactly 1, so standardized and ordinarily scaled
    data are untouched, and one beyond it is corrected only for the excess. The
    shrinkage rather than a cutoff matters: a cutoff gives two columns of nearly the
    same scale entries 4⁴ apart, and on a six-class problem with features around 20
    that took convergence past 100 iterations where the shrunk version needs 76
    (scikit-learn: 251). On breast_cancer, raw units, it converges in 72 where
    scikit-learn needs 3267. The intercept's own entry is always 1.
    """
    curvature = scale * scale + 4.0 * l2
    exponents = np.zeros_like(curvature)
    # A column that overflowed float32 when centered has no usable scale; its
    # objective is not finite either, and the solver reports that.
    usable = np.isfinite(curvature) & (curvature > 0.0)
    exponents[usable] = np.round(0.5 * np.log2(curvature[usable]))
    exponents = np.sign(exponents) * np.maximum(np.abs(exponents) - _PRECONDITION_DEADBAND, 0.0)
    weights = np.ldexp(1.0, (-2 * exponents).astype(np.int64))
    if fit_intercept:
        weights = np.append(weights, 1.0)
    return np.tile(weights, (n_targets, 1))


def _weight_scale(sample_weight: np.ndarray) -> float:
    """The power of two that brings ``max(sample_weight)`` into ``[1, 2)``.

    Dividing by a power of two is exact, so unit and ordinary weights are left
    bitwise identical and only the representation of extreme ones changes.
    """
    largest = float(np.max(sample_weight))
    return math.ldexp(1.0, math.frexp(largest)[1] - 1)


def solve_logistic(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_classes: int,
    sample_weight: np.ndarray | None = None,
    config: LogisticConfig,
    callback=None,
) -> LogisticSolution:
    """Fit penalized logistic regression by L-BFGS.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Dense and finite, float64 or float32. Centered in float64 when there is an
        intercept, then rounded to float32 for computation.
    y : ndarray of shape (n_samples,)
        Integer class codes in ``[0, n_classes)``. Encoding labels is the caller's
        job; this function never sees the original labels.
    n_classes : int
        Two selects the binary objective — one weight vector for code 1 — and more
        selects the unconstrained multinomial objective.
    sample_weight : ndarray of shape (n_samples,), optional
        float64, non-negative, with any class weights already folded in. ``None``
        means unweighted, and takes a code path that never multiplies by ones.
    config : LogisticConfig
    callback : callable, optional
        Called as ``callback(coef, intercept)`` after every accepted L-BFGS step,
        with float64 arrays in the caller's units — the same shapes and meaning as
        :attr:`LogisticSolution.coef` and ``.intercept``.

    Returns
    -------
    LogisticSolution

    Notes
    -----
    The starting point is always zero. There is no warm start and no cache: the
    ancestor's continuation cache was keyed on the *address* of the training array
    and enabled by default, which made ``fit`` depend on what had been fitted
    earlier in the same process. A fit here is a pure function of its arguments.
    """
    import mlx.core as mx

    if n_classes < 2:
        raise ValueError(f"n_classes must be at least 2, got {n_classes}")

    y_host = np.asarray(y)
    n_samples = np.shape(X)[0]
    if y_host.shape != (n_samples,):
        raise ValueError(
            f"y must have shape ({n_samples},) to match X, got {y_host.shape}"
        )

    X_host, center, scale = _prepare_design(X, config.fit_intercept)
    n_features = X_host.shape[1]

    if sample_weight is None:
        sw_sum = float(n_samples)
        weights = None
        normalizer = sw_sum
    else:
        weights = np.asarray(sample_weight, dtype=np.float64)
        sw_sum = float(np.sum(weights))
        if sw_sum <= 0.0:
            # The wording matters: scikit-learn 1.9's
            # check_all_zero_sample_weights_error matches r"weight.*zero|zero.*weight",
            # and every estimator is expected to say so recognizably.
            raise ValueError(
                "Sample weights sum to zero: every sample weight is zero, so there "
                "is nothing to fit."
            )
        weight_scale = _weight_scale(weights)
        weights = weights / weight_scale
        normalizer = sw_sum / weight_scale

    # From the unscaled sum: the penalty is 1/(C·sw_sum) whatever the weights'
    # representation.
    l2 = _l2_reg_strength(config.C, sw_sum)
    n_targets = 1 if n_classes == 2 else n_classes
    n_columns = n_features + int(config.fit_intercept)

    def unpack(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        coef = np.array(theta[:, :n_features], dtype=np.float64, order="C")
        if config.fit_intercept:
            intercept = np.asarray(theta[:, n_features], dtype=np.float64)
            if center is not None:
                # Undo the centering exactly: b = b′ − w·μ, in float64, with the μ
                # that was actually subtracted.
                intercept = intercept - coef @ center
        else:
            intercept = np.zeros(n_targets, dtype=np.float64)
        return coef, np.ascontiguousarray(intercept)

    iteration_callback = None
    if callback is not None:

        def iteration_callback(theta):
            callback(*unpack(theta))

    with use_device(config.device) as kind:
        Xm = mx.array(X_host, dtype=mx.float32)
        swm = None if weights is None else mx.array(weights.astype(NUMPY_FLOAT))

        if n_targets == 1:
            transposed = kind == "gpu"
            if transposed:
                # Transpose once on the device and keep only the transpose, so the
                # resident footprint is unchanged; see the module docstring.
                Xm = mx.contiguous(Xm.T)
                mx.eval(Xm)
            fun = _binary_objective(
                mx,
                Xm,
                mx.array(y_host.astype(NUMPY_FLOAT), dtype=mx.float32),
                swm,
                normalizer,
                l2,
                config.fit_intercept,
                n_features,
                transposed=transposed,
            )
        else:
            codes = y_host.astype(np.int32)
            fun = _multinomial_objective(
                mx,
                Xm,
                mx.array(codes.reshape(-1, 1), dtype=mx.int32),
                mx.array(
                    np.eye(n_classes, dtype=NUMPY_FLOAT)[codes], dtype=mx.float32
                ),
                swm,
                normalizer,
                l2,
                config.fit_intercept,
                n_features,
            )

        theta0 = mx.zeros((n_targets, n_columns), dtype=mx.float32)
        state = minimize(
            fun,
            theta0,
            LBFGSConfig(
                max_iter=config.max_iter,
                tol=config.tol,
                history_size=config.history_size,
                max_line_search=config.max_line_search,
            ),
            callback=iteration_callback,
            preconditioner=_preconditioner(scale, l2, n_targets, config.fit_intercept),
        )
        theta = np.asarray(state.x, dtype=NUMPY_FLOAT).astype(np.float64)

    coef, intercept = unpack(theta)
    return LogisticSolution(
        coef=coef,
        intercept=intercept,
        n_iter=state.n_iter,
        objective=state.objective,
        grad_max_abs=state.grad_max_abs,
        converged=state.converged,
        stop_reason=state.stop_reason,
        n_fev=state.n_fev,
        initial_grad_max_abs=state.initial_grad_max_abs,
    )


# --------------------------------------------------------------------- reference


def reference_objective(
    coef: np.ndarray,
    intercept: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_classes: int,
    C: float = 1.0,
    sample_weight: np.ndarray | None = None,
) -> float:
    """The same objective in float64 NumPy, for parity assertions.

    ``docs/fp32_policy.md`` §3 makes the objective value the *strong* parity
    assertion and the coefficients the weak one, because two solvers can reach
    materially different coefficients that are equally optimal on an
    ill-conditioned problem. That assertion is only meaningful if both coefficient
    sets are scored by one function at one precision — this one.

    Parameters mirror :func:`solve_logistic`; ``coef`` is ``(n_targets, n_features)``
    with ``n_targets == 1`` for binary.

    Examples
    --------
    >>> import numpy as np
    >>> from mlxlearn._solvers.logistic import reference_objective
    >>> X = np.array([[0.0], [1.0], [2.0], [3.0]])
    >>> y = np.array([0, 0, 1, 1])
    >>> float(round(reference_objective(np.zeros((1, 1)), np.zeros(1), X, y,
    ...                                 n_classes=2, C=1.0), 12))
    0.69314718056
    """
    Xd = np.asarray(X, dtype=np.float64)
    codes = np.asarray(y).astype(np.int64)
    W = np.atleast_2d(np.asarray(coef, dtype=np.float64))
    b = np.asarray(intercept, dtype=np.float64).ravel()

    n_samples = Xd.shape[0]
    if sample_weight is None:
        sw = None
        sw_sum = float(n_samples)
    else:
        sw = np.asarray(sample_weight, dtype=np.float64)
        sw_sum = float(sw.sum())

    raw = Xd @ W.T + b
    if n_classes == 2:
        score = raw[:, 0]
        target = codes.astype(np.float64)
        loss = np.logaddexp(0.0, score) - target * score
    else:
        shift = raw.max(axis=1, keepdims=True)
        log_norm = shift[:, 0] + np.log(np.exp(raw - shift).sum(axis=1))
        loss = log_norm - raw[np.arange(n_samples), codes]

    weighted = loss if sw is None else loss * sw
    penalty = 0.0 if math.isinf(C) else 0.5 * float((W * W).sum()) / (C * sw_sum)
    return float(weighted.sum() / sw_sum + penalty)
