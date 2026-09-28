"""Blocked SVM kernel evaluation in MLX.

Four kernels — ``linear``, ``poly``, ``rbf``, ``sigmoid`` — with scikit-learn's
``gamma`` resolution, evaluated one row block at a time so that the working set on
the device is bounded by a byte budget rather than by ``n_samples²``. Nothing here
knows what an estimator is; everything takes and returns arrays and frozen
plain-data objects.

Three things this module is responsible for.

**Never requiring a full Gram.** :class:`KernelRows` answers "give me rows
``i, j, …``" from device-resident training data, in one round trip, and
:func:`kernel_combine` folds ``K(A, B) @ W`` block by block. The dense ``n × n``
matrix is materialized only when a caller explicitly asks for it via
:func:`kernel_matrix`, which is what an SMO fit does when — and only when — its
row-cache budget already permits holding every row.

**Exactness under float32.** The squared distance inside the RBF kernel comes from
the expanded identity ``‖a‖² + ‖b‖² − 2⟨a, b⟩``, which is hostile in float32 for
data far from the origin (``docs/fp32_policy.md`` §2.1). Distances are
translation-invariant, so both matrices are translated by a common ``center``
before the identity is applied — exact mathematics, and it is what keeps the
cancellation at the variance scale instead of the coordinate scale. Centering is
applied *only* to the RBF kernel: ``linear``, ``poly`` and ``sigmoid`` read the raw
inner product, which is not translation-invariant, and shifting it would compute a
different kernel. :func:`kernel_frame` does the translation in float64 *before*
the cast, and adds an exact power-of-two scale that :func:`frame_spec` folds back
into ``gamma``, so neither a large offset nor an extreme magnitude reaches float32.

**Integer powers by repeated squaring.** ``poly`` is ``(γ⟨a,b⟩ + coef0)ᵈ`` with an
integer degree. Evaluating that as a floating-point power returns NaN whenever the
base is negative, which happens routinely for ``coef0=0`` and a negative inner
product. LIBSVM uses binary exponentiation and so does this module, so a negative
base raised to an odd degree yields the negative value it should.
"""

from __future__ import annotations

import functools
import threading
from dataclasses import dataclass, replace

import numpy as np

from .._common.arrays import NUMPY_FLOAT
from .._common.config import get_tuning
from .._common.device import inference_device, use_device
from ..exceptions import UnsupportedInputError, UnsupportedParameterError

__all__ = [
    "SUPPORTED_KERNELS",
    "KernelSpec",
    "KernelRows",
    "resolve_gamma",
    "kernel_matrix",
    "kernel_combine",
    "kernel_diagonal",
]

#: The kernels mlxlearn evaluates. ``"precomputed"`` and callables are absent on
#: purpose: they are not slower here, they are simply not this module's job, and a
#: caller must be told so rather than handed a different kernel's answer.
SUPPORTED_KERNELS = ("linear", "poly", "rbf", "sigmoid")


@dataclass(frozen=True, slots=True)
class KernelSpec:
    """A fully resolved kernel: no ``"scale"``, no ``"auto"``, no estimator.

    Attributes
    ----------
    kernel : {"linear", "poly", "rbf", "sigmoid"}
    gamma : float
        Already resolved by :func:`resolve_gamma`. Unused by ``"linear"``.
    degree : int
        Integer polynomial degree. Unused unless ``kernel="poly"``.
    coef0 : float
        Independent term. Unused by ``"linear"`` and ``"rbf"``.
    """

    kernel: str
    gamma: float = 1.0
    degree: int = 3
    coef0: float = 0.0

    def __post_init__(self) -> None:
        if self.kernel not in SUPPORTED_KERNELS:
            raise UnsupportedParameterError(
                f"kernel={self.kernel!r} is not evaluated by mlxlearn. Supported "
                f"kernels are: {', '.join(repr(k) for k in SUPPORTED_KERNELS)}.",
                parameter="kernel",
                value=self.kernel,
            )
        if int(self.degree) < 0:
            raise ValueError(f"degree must be non-negative, got {self.degree!r}")

    @property
    def uses_center(self) -> bool:
        """Whether a common translation may be applied before evaluation.

        True only for ``"rbf"``, whose squared-distance argument is
        translation-invariant. Translating the inputs of any other kernel here
        would silently evaluate a different kernel.
        """
        return self.kernel == "rbf"


def resolve_gamma(gamma, X: np.ndarray, *, kernel: str | None = None) -> float:
    """Resolve ``gamma`` the way scikit-learn resolves it.

    Parameters
    ----------
    gamma : {"scale", "auto"} or float
    X : ndarray of shape (n_samples, n_features)
        The validated training matrix, float64. The variance is taken over *all*
        entries — ``X.var()``, not a per-feature variance — because that is what
        scikit-learn does and ``gamma`` has to mean the same number here.
    kernel : str, optional
        When ``"linear"``, a numeric ``gamma`` is returned as given, including
        ``0.0``: the linear kernel never reads it, and scikit-learn stores it
        unvalidated for exactly that reason.

    Returns
    -------
    float

    Notes
    -----
    ``"scale"`` is ``1 / (n_features * X.var())``, falling back to ``1.0`` only when
    the variance is exactly zero — scikit-learn's ``X_var != 0``, not a threshold.
    This was ``variance > 1e-12``, which quietly resolved ``gamma=1`` for any data
    whose features sit around 1e-6 or below, where the correct value is 1e11 and
    up: a kernel that is 1 everywhere and a model at chance, with no warning.
    ``"auto"`` is ``1 / n_features``.

    The variance is computed on the float64 validated matrix, not on the float32
    copy mlxlearn computes with. A float32 variance differs in the ninth
    significant digit, and that difference would propagate into every kernel entry
    of every fit — a systematic offset from scikit-learn bought for nothing.

    The result may lie outside float32's range — ``"scale"`` on data around 1e-20
    gives about 1e38. That is not an error here: :func:`frame_spec` folds the
    power-of-two scale of the compute frame into it, and only the folded value is
    ever evaluated in float32.

    Examples
    --------
    >>> import numpy as np
    >>> from mlxlearn._kernels.kernels import resolve_gamma
    >>> X = np.array([[0.0, 1.0], [2.0, 3.0]])
    >>> resolve_gamma("auto", X)
    0.5
    >>> resolve_gamma(0.25, X)
    0.25
    """
    if isinstance(gamma, str):
        if gamma == "scale":
            variance = float(X.var())
            return 1.0 / (X.shape[1] * variance) if variance != 0.0 else 1.0
        if gamma == "auto":
            return 1.0 / X.shape[1]
        raise UnsupportedParameterError(
            f"gamma={gamma!r} is not a value mlxlearn resolves. Use 'scale', 'auto', "
            "or a positive float.",
            parameter="gamma",
            value=gamma,
        )
    value = float(gamma)
    if kernel == "linear":
        return value
    if value <= 0.0:
        # scikit-learn's constraint is Interval(Real, 0.0, None, closed="left"),
        # so gamma=0.0 is legal there and produces a degenerate but well-defined
        # model. mlxlearn's kernels do not implement it — and because
        # scikit-learn *can* serve the request, this must be a CapabilityError so
        # that patched mode falls back instead of raising. A plain ValueError here
        # propagated straight through mlx_guard and broke Layer 2 too.
        raise UnsupportedParameterError(
            f"mlxlearn does not implement gamma={value!r}; the MLX kernels require a "
            "strictly positive gamma.",
            parameter="gamma",
            value=value,
            advice="scikit-learn accepts it, so under patch_sklearn() this falls back "
            "rather than failing.",
        )
    return value


def kernel_diagonal(spec: KernelSpec, X: np.ndarray) -> np.ndarray:
    """``K(x, x)`` for every row, computed in closed form, float64.

    Reading the diagonal off a materialized Gram would force one to exist; the
    closed forms below do not. They are also what LIBSVM uses, and it evaluates
    them in double while holding the cached kernel rows in single — the same split
    mlxlearn uses, which is why the two agree on the working-set quadratic
    coefficient to better than the tolerance the solver stops on.
    """
    if spec.kernel == "rbf":
        # ‖x − x‖² = 0 exactly, so exp(0) = 1 exactly. Computing it would only
        # introduce a rounding error the analytic value does not have.
        return np.ones(X.shape[0], dtype=np.float64)

    squared = np.einsum("ij,ij->i", X, X, dtype=np.float64)
    if spec.kernel == "linear":
        return squared
    z = spec.gamma * squared + spec.coef0
    if spec.kernel == "sigmoid":
        return np.tanh(z)
    return _host_integer_power(z, int(spec.degree))


def _host_integer_power(base: np.ndarray, degree: int) -> np.ndarray:
    """``base ** degree`` by repeated squaring, correct for negative bases."""
    if degree == 0:
        return np.ones_like(base)
    result: np.ndarray | None = None
    factor = base
    exponent = degree
    while exponent:
        if exponent & 1:
            result = factor.copy() if result is None else result * factor
        exponent >>= 1
        if exponent:
            factor = factor * factor
    assert result is not None
    return result


# --------------------------------------------------------------------------------------
# the float32 compute frame
# --------------------------------------------------------------------------------------

#: float32's largest finite value. Restated rather than imported from
#: ``_common.validation``, which imports scikit-learn; this package must not.
_FLOAT32_MAX = float(np.finfo(np.float32).max)

#: Elements per host block when translating rows in float64: small enough that the
#: float64 buffer stays cache-resident, however large the input is.
_FRAME_BLOCK_ELEMENTS = 1 << 16


def kernel_frame(spec: KernelSpec, X: np.ndarray) -> tuple[np.ndarray | None, float]:
    """The exact map ``x ↦ (x − center) · scale`` under which ``spec`` runs in float32.

    Two things float32 cannot absorb are removed by exact arithmetic in float64,
    *before* the cast, rather than hoped away after it.

    **Offset** (``rbf`` only, the one translation-invariant kernel). The float64
    training mean is subtracted first. Casting first — what this module used to do —
    rounds away exactly the digits that distinguish samples: Unix timestamps within
    one hour collapse to 29 distinct float32 values out of 5 000, and no later
    translation can bring them back.

    **Magnitude.** An exact power of two puts the largest mapped entry in
    ``[0.5, 1)``, so squares and inner products can neither overflow (centred
    ``|x| ≳ 1.8e19 / √d``) nor sink into float32's subnormal range, which the GPU
    flushes to zero. :func:`frame_spec` folds the scale back into ``gamma``, so the
    kernel is unchanged. Scaling by a power of two commutes with float32 rounding,
    so for data already in range every intermediate rounds exactly as it did
    unscaled — the result is bit-identical, not merely close.

    ``"linear"`` gets the identity: its value *is* the raw inner product, with no
    ``gamma`` to absorb a scale, so :func:`check_frame_range` guards it instead.

    Parameters
    ----------
    spec : KernelSpec
    X : ndarray of shape (n_samples, n_features)
        The validated training matrix, float64.

    Returns
    -------
    center : ndarray of shape (n_features,), float64, or None
    scale : float
        An exact power of two; ``1.0`` when there is nothing to scale.
    """
    X = np.asarray(X, dtype=np.float64)
    if spec.kernel == "linear" or X.size == 0:
        return None, 1.0
    if spec.uses_center:
        center = X.mean(axis=0)
        # Rounded subtraction is monotone, so the largest |x − c| is attained at a
        # column's max or min: two reductions and no n × d temporary.
        extent = max(
            float(np.max(X.max(axis=0) - center)), float(np.max(center - X.min(axis=0)))
        )
    else:
        center = None
        extent = max(abs(float(X.max())), abs(float(X.min())))
    return center, _power_of_two_scale(extent)


def _power_of_two_scale(extent: float) -> float:
    """``2**-e`` with ``extent = m · 2**e``, ``m ∈ [0.5, 1)``; ``1.0`` if degenerate."""
    if not (0.0 < extent < np.inf):
        return 1.0
    _, exponent = np.frexp(extent)
    scale = float(np.ldexp(1.0, -int(exponent)))
    # Below 2**-1022 the reciprocal is not a finite double. Data that small is zero
    # in float32 either way, and so is every kernel distance LIBSVM would compute.
    return scale if 0.0 < scale < np.inf else 1.0


def frame_spec(spec: KernelSpec, scale: float) -> KernelSpec:
    """``spec`` with the frame's scale folded into ``gamma``: ``γ' = γ / scale²``.

    ``γ'‖s·a − s·b‖² = γ‖a − b‖²`` and ``γ'⟨s·a, s·b⟩ = γ⟨a, b⟩`` exactly, so the kernel
    evaluated on mapped inputs is the kernel of the original ones. This is also what
    lets a ``gamma`` outside float32's range be served: ``"scale"`` on data around
    1e-20 resolves to about 4e38, and folds back to order one.

    Raises
    ------
    UnsupportedParameterError
        When the folded ``gamma``, or ``coef0``, is not finite in float32 — an
        exponent the device cannot represent. scikit-learn evaluates the kernel in
        float64 and can serve it, so this is a capability limit, not bad input.
    """
    if spec.kernel == "linear":
        return spec
    # Two divisions rather than one by scale**2, which can overflow for tiny data.
    gamma = spec.gamma / scale / scale
    if not abs(gamma) <= _FLOAT32_MAX:
        raise UnsupportedParameterError(
            f"gamma={spec.gamma!r} is {gamma:.3g} once folded into mlxlearn's float32 "
            "compute frame, which float32 cannot represent.",
            parameter="gamma",
            value=spec.gamma,
            advice="scikit-learn evaluates the kernel in float64 and can serve it, so "
            "under patch_sklearn() this falls back rather than failing.",
        )
    if not abs(spec.coef0) <= _FLOAT32_MAX:
        raise UnsupportedParameterError(
            f"coef0={spec.coef0!r} is not finite in float32, where mlxlearn evaluates "
            "the kernel.",
            parameter="coef0",
            value=spec.coef0,
            advice="scikit-learn evaluates it in float64, so under patch_sklearn() this "
            "falls back rather than failing.",
        )
    return replace(spec, gamma=gamma)


def apply_frame(X, center: np.ndarray | None, scale: float) -> np.ndarray:
    """``(X − center) · scale``, rounded to float32 once, as a new array.

    The translation is computed in float64 — that is its whole point — through a
    small reused buffer rather than a float64 copy of ``X``. The power-of-two scale
    is exact in either precision, so it is folded into the final store. Every entry
    is mapped independently, so the blocking cannot change a single bit. The output
    never aliases ``X``. Entries that land beyond float32 become ``inf`` rather than
    raising here; :func:`check_frame_range` is what turns that into an error.
    """
    X = np.asarray(X)
    out = np.empty(X.shape, dtype=NUMPY_FLOAT)
    if X.size == 0:
        return out
    with np.errstate(over="ignore"):
        if center is None:
            if scale == 1.0:
                np.copyto(out, X, casting="same_kind")
            else:
                np.multiply(X, scale, out=out)
            return out
        step = max(1, _FRAME_BLOCK_ELEMENTS // max(int(X.shape[1]), 1))
        buffer = np.empty((min(step, X.shape[0]), X.shape[1]), dtype=np.float64)
        for start in range(0, X.shape[0], step):
            stop = min(start + step, X.shape[0])
            block = buffer[: stop - start]
            np.subtract(X[start:stop], center, out=block)
            np.multiply(block, scale, out=out[start:stop])
    return out


def check_frame_range(
    spec: KernelSpec, X32: np.ndarray, *, name: str = "X", fallback_advice: bool = True
) -> None:
    """Raise when float32 kernel arithmetic on mapped rows ``X32`` could overflow.

    ``rbf`` evaluates ``‖a‖² + ‖b‖² − 2⟨a, b⟩``, at most four times the largest
    squared row norm; the other kernels evaluate ``⟨a, b⟩``, bounded by it. Within
    the bound no partial sum can overflow either (Cauchy–Schwarz on each prefix).

    After :func:`kernel_frame` the training rows are always in range. What this
    catches is a linear-kernel fit with ``‖x‖² > 3.4e38``, and an inference query
    so far outside the training extent that the training-derived scale cannot hold
    it. Values past ``3.4e38`` themselves are rejected earlier, on the raw input.
    ``fallback_advice=False`` leaves out the ``patch_sklearn()`` advice, which only
    a fit can act on.

    Raises
    ------
    UnsupportedInputError
        With ``reason="float32-overflow"``, so patched mode can fall back at fit.
    """
    if X32.size == 0:
        return
    factor = 4.0 if spec.kernel == "rbf" else 1.0
    # ‖x‖² ≤ d · max|x|² costs two reductions and no temporary; the exact norms are
    # computed only when that cheap bound cannot rule an overflow out.
    peak = max(float(X32.max()), -float(X32.min()))
    if factor * X32.shape[1] * peak * peak <= _FLOAT32_MAX:
        return
    largest = float(np.einsum("ij,ij->i", X32, X32, dtype=np.float64).max())
    if not factor * largest <= _FLOAT32_MAX:
        raise UnsupportedInputError(
            f"{name} has a row whose squared norm is {largest:.3g} in mlxlearn's float32 "
            f"compute frame, so the {spec.kernel!r} kernel would overflow float32 (the "
            f"limit is {_FLOAT32_MAX:.3g}) and return infinity or NaN. Rescale the "
            "data.",
            reason="float32-overflow",
            advice=(
                "Or run under patch_sklearn(), which hands a fit on such data to "
                "scikit-learn's float64 path."
            )
            if fallback_advice
            else "",
        )


class KernelBasis:
    """A fixed right-hand kernel operand — an SVC's support vectors — kept on device.

    Every ``predict`` used to re-map, re-upload and re-reduce the support vectors:
    0.2–0.4 ms of fixed cost at 4 000 × 256, against about 0.25 ms for scikit-learn's
    *entire* single-row predict. This does that work once.

    Owned by the fitted estimator and released with it — never a cache keyed on the
    identity of an array. It holds no host copy of its own: the device tensors are
    built on first use from the rows the owner passes in, and dropped on pickle, so a
    pickled model carries its support vectors exactly once and can be loaded where
    the device tensors could not have been. The build is locked, so concurrent first
    predictions build it once.

    Attributes
    ----------
    spec : KernelSpec
        The *compute* spec, with the frame's scale already folded in by
        :func:`frame_spec`.
    center, scale
        The frame, from :func:`kernel_frame`.
    """

    __slots__ = ("spec", "center", "scale", "_lock", "_operand")

    def __init__(
        self, spec: KernelSpec, *, center: np.ndarray | None = None, scale: float = 1.0
    ) -> None:
        self.spec = spec
        self.center = None if center is None else np.array(center, dtype=np.float64)
        self.scale = float(scale)
        self._lock = threading.Lock()
        self._operand: tuple | None = None  # (mx.array (n, d), mx.array (n,) or None)

    def __getstate__(self) -> dict:
        """Pickle the frame only; the device tensors are derived from the owner's rows."""
        return {"spec": self.spec, "center": self.center, "scale": self.scale}

    def __setstate__(self, state: dict) -> None:
        self.spec = state["spec"]
        self.center = state["center"]
        self.scale = state["scale"]
        self._lock = threading.Lock()
        self._operand = None

    def prepare(self, X, *, name: str = "X", fallback_advice: bool = True) -> np.ndarray:
        """``X`` mapped into this basis's float32 frame, range-checked."""
        mapped = apply_frame(X, self.center, self.scale)
        check_frame_range(self.spec, mapped, name=name, fallback_advice=fallback_advice)
        return mapped

    def _device_operand(self, rows: np.ndarray, device: str) -> tuple:
        operand = self._operand
        if operand is not None:
            return operand
        with self._lock:
            if self._operand is None:
                import mlx.core as mx

                prepared = self.prepare(rows, name="support_vectors_", fallback_advice=False)
                with use_device(device):
                    right = mx.array(prepared, dtype=mx.float32)
                    if self.spec.kernel == "rbf":
                        right_sq = mx.sum(right * right, axis=1)
                        mx.eval(right, right_sq)
                    else:
                        right_sq = None
                        mx.eval(right)
                # Published whole, in one assignment, so an unlocked reader sees
                # either nothing or a complete operand.
                self._operand = (right, right_sq)
            return self._operand

    def combine(
        self,
        Q,
        rows: np.ndarray,
        weights: np.ndarray,
        *,
        backend: str,
        block_rows: int | None = None,
    ) -> np.ndarray:
        """``K(Q, rows) @ weights`` for raw queries ``Q``, float64.

        Parameters
        ----------
        Q : ndarray of shape (n_queries, n_features)
            Unmapped queries, float64 or float32; mapped here.
        rows : ndarray of shape (n_rows, n_features)
            The owner's unmapped basis rows. Read only when the device operand has
            not been built yet.
        weights : ndarray of shape (n_rows, n_outputs)
        backend : {"mlx", "cpu"}

        Notes
        -----
        The same tile loop as :func:`kernel_combine` — float32 tiles on the device,
        pipelined against a float64 contraction on the host — so the two cannot
        drift apart; see there for the precision split and for what ``block_rows``
        can change.
        """
        import mlx.core as mx

        # Inference: the fitted backend is sticky, so there is no fallback to advise.
        Q32 = self.prepare(Q, fallback_advice=False)
        W = np.asarray(weights, dtype=np.float64)
        out = np.zeros((Q32.shape[0], W.shape[1]), dtype=np.float64)
        if Q32.shape[0] == 0 or rows.shape[0] == 0:
            return out

        device = inference_device(backend)
        right, right_sq = self._device_operand(rows, device)
        with use_device(device):
            _combine_tiles(mx, self.spec, device, Q32, right, right_sq, W, out, block_rows)
        return out


# --------------------------------------------------------------------------------------
# device evaluation
# --------------------------------------------------------------------------------------


def _block_rows(n_rows: int, n_cols: int, device: str, block_rows: int | None) -> int:
    """How many rows of the left operand to evaluate at once on ``device``.

    Derived from the same byte budget the distance kernel uses, halved because a
    block needs both the inner-product tile and the transformed tile alive at the
    same time. A fixed constant would be wrong by an order of magnitude on both an
    8 GB laptop and a 128 GB desktop.
    """
    if block_rows is not None:
        return max(1, min(int(block_rows), max(n_rows, 1)))
    tuning = get_tuning()
    budget = (
        tuning.distance_block_bytes if device == "gpu" else tuning.cpu_distance_block_bytes
    )
    rows = int(budget // max(8 * n_cols, 1))
    return max(1, min(max(n_rows, 1), rows))


def _integer_power(mx, base, degree: int):
    """``base ** degree`` on device, by repeated squaring.

    ``mx.power`` with a floating exponent returns NaN for a negative base, and a
    negative base is the normal case for ``coef0=0`` with an obtuse angle between
    two samples. Repeated squaring is also exactly what LIBSVM's ``powi`` does, so
    the rounding sequence matches.
    """
    if degree == 0:
        return mx.ones_like(base)
    result = None
    factor = base
    exponent = degree
    while exponent:
        if exponent & 1:
            result = factor if result is None else result * factor
        exponent >>= 1
        if exponent:
            factor = factor * factor
    return result


def _transform(mx, spec: KernelSpec, dot, a_sq, b_sq):
    """Turn an inner-product tile into a kernel tile, on device."""
    if spec.kernel == "linear":
        return dot
    if spec.kernel == "rbf":
        squared = a_sq[:, None] + b_sq[None, :] - 2.0 * dot
        # Clamp before the exponential: the identity can produce a small negative
        # value for two nearly identical rows, and exp(-gamma * negative) would
        # return a kernel entry greater than one.
        return mx.exp(-spec.gamma * mx.maximum(squared, 0.0))
    z = spec.gamma * dot + spec.coef0
    if spec.kernel == "sigmoid":
        return mx.tanh(z)
    return _integer_power(mx, z, int(spec.degree))


@functools.lru_cache(maxsize=1)
def _fused_rbf():
    """The rbf branch of :func:`_transform` compiled into one GPU kernel.

    Evaluated eagerly, that branch is six full passes over the tile — broadcast
    add, scale, subtract, clamp, scale, exponentiate — each writing an ``n_a × n_b``
    intermediate. Fused, it is one pass, which measured 1.5–5.7× faster per tile.

    ``gamma`` is an *argument*, a 0-d float32 array, and never a constant captured
    in the traced function. Captured, MLX embeds its own rounding of the Python
    float into the kernel, which differs from the eager path's float32 scalar by
    an ulp for most values — including the default ``gamma="scale"`` — and every
    kernel entry then moves with it. Passed as an array, the fused kernel is
    bit-identical to :func:`_transform`, and it is one compiled function for every
    ``gamma`` rather than one per fit, so this is a cache of *code*, keyed on
    nothing: the "no cache keyed on data" rule is about the other kind.

    ``shapeless=True`` because tile shapes vary with every query batch, and there
    is nothing shape-dependent in the graph to specialize on.
    """
    import mlx.core as mx

    def rbf(dot, a_sq, b_sq, neg_gamma):
        squared = a_sq[:, None] + b_sq[None, :] - 2.0 * dot
        return mx.exp(neg_gamma * mx.maximum(squared, 0.0))

    return mx.compile(rbf, shapeless=True)


def _tile_transform(mx, spec: KernelSpec, device: str):
    """``f(dot, a_sq, b_sq)`` turning an inner-product tile into a kernel tile.

    The fused rbf kernel is used on the GPU only — the device the work runs on,
    not the backend it was fitted for. On MLX's CPU device the same compiled
    graph measured 2.3–2.6× *slower* than the eager ops, and its first call per
    process shells out to a C compiler for about a second — neither of which the
    CPU path, which Layer 1 takes for every problem below the SVC crossover, nor
    a GPU-fitted model loaded on a GPU-less host, should pay for an identical
    answer.
    """
    if spec.kernel == "rbf" and device == "gpu":
        fused = _fused_rbf()
        neg_gamma = mx.array(-spec.gamma, dtype=mx.float32)
        return lambda dot, a_sq, b_sq: fused(dot, a_sq, b_sq, neg_gamma)
    return lambda dot, a_sq, b_sq: _transform(mx, spec, dot, a_sq, b_sq)


#: Rows of a kernel tile contracted per host matmul in :func:`kernel_combine`.
#: Converting a whole tile to float64 first materializes a twin twice its size
#: (256 MiB for a full-budget tile) and streams it through memory once more than
#: necessary; a few hundred rows at a time keep the float64 copy in cache, and are
#: still enough rows for BLAS to run its full-width kernel.
_HOST_CONTRACT_ROWS = 256


def _contract_tile(tile, weights: np.ndarray, out: np.ndarray) -> None:
    """``out[:] = tile @ weights`` in float64, a chunk of rows at a time.

    Blocks until ``tile`` has been evaluated — the only synchronization point in
    the pipelined loops below.
    """
    host = np.asarray(tile)
    for start in range(0, host.shape[0], _HOST_CONTRACT_ROWS):
        stop = start + _HOST_CONTRACT_ROWS
        out[start:stop] = host[start:stop].astype(np.float64) @ weights


def _combine_pipelined(mx, transform, n_a: int, rows: int, left, right, b_sq, weights, out):
    """Fold ``K(left, right) @ weights`` into ``out``, overlapping device and host.

    ``left(start, stop)`` returns one block of the left operand and its squared
    norms (``None`` unless rbf). Tile ``i + 1`` is submitted with ``mx.async_eval``
    *before* the host contracts tile ``i``, so the device computes the next tile
    while the host is busy with this one; evaluated serially, the GPU sat idle for
    the contraction, measured at 39–47% of multiclass predict time.

    Memory: at most three tiles are alive on the device — the one being contracted
    and the next one's inner product and result — where the serial loop held two
    plus a float64 host twin of one, i.e. the equivalent of four. The chunked
    contraction is what pays for the pipelining, so the block size derived from
    the byte budget does not have to shrink.
    """
    right_t = right.T

    def submit(start):
        block, a_sq = left(start, min(start + rows, n_a))
        tile = transform(block @ right_t, a_sq, b_sq)
        mx.async_eval(tile)
        return tile

    tile = submit(0)
    for start in range(0, n_a, rows):
        following = submit(start + rows) if start + rows < n_a else None
        _contract_tile(tile, weights, out[start : start + rows])
        # Rebinding releases this tile before the next submission, which is what
        # keeps the device at three live tiles rather than four.
        tile = following


def _combine_tiles(mx, spec, device, left32, right, right_sq, weights, out, block_rows) -> None:
    """``out[:] = K(left32, right) @ weights``: the one tile loop behind every combine.

    ``left32`` is a host float32 matrix already in the kernel's frame; ``right``
    and ``right_sq`` (its squared row norms, rbf only) are on the device. Blocked
    by :func:`_block_rows` for ``device``, transformed by :func:`_tile_transform`
    and pipelined by :func:`_combine_pipelined`, inside the caller's
    :func:`~mlxlearn._common.device.use_device` block for that device.
    """
    n_left = int(left32.shape[0])
    rows = _block_rows(n_left, int(right.shape[0]), device, block_rows)
    needs_norms = spec.kernel == "rbf"

    def left(start, stop):
        block = mx.array(left32[start:stop], dtype=mx.float32)
        return block, (mx.sum(block * block, axis=1) if needs_norms else None)

    transform = _tile_transform(mx, spec, device)
    _combine_pipelined(mx, transform, n_left, rows, left, right, right_sq, weights, out)


def _prepare(spec: KernelSpec, A, center) -> np.ndarray:
    """Host-side float32 copy of ``A``, translated when the kernel allows it."""
    out = np.ascontiguousarray(A, dtype=NUMPY_FLOAT)
    if center is not None and spec.uses_center:
        out = np.ascontiguousarray(out - np.asarray(center, dtype=NUMPY_FLOAT))
    return out


def kernel_matrix(
    spec: KernelSpec,
    A: np.ndarray,
    B: np.ndarray | None = None,
    *,
    backend: str = "cpu",
    center: np.ndarray | None = None,
    block_rows: int | None = None,
) -> np.ndarray:
    """The dense kernel matrix ``K(A, B)``, float32.

    Parameters
    ----------
    spec : KernelSpec
    A : ndarray of shape (n_a, n_features)
    B : ndarray of shape (n_b, n_features), optional
        Defaults to ``A``.
    backend : {"mlx", "cpu"}, default="cpu"
        Which MLX device evaluates the blocks.
    center : ndarray of shape (n_features,), optional
        Common translation, applied only for ``kernel="rbf"``.
    block_rows : int, optional
        Rows of ``A`` per block. Defaults to a memory-derived size.

    Returns
    -------
    ndarray of shape (n_a, n_b), float32

    Notes
    -----
    This is the one function here that does materialize an ``n_a × n_b`` result, so
    callers who cannot afford that should use :func:`kernel_combine` or
    :class:`KernelRows` instead. The *evaluation* is still blocked: the device
    never holds more than one block at a time regardless of the output size.
    """
    import mlx.core as mx

    A32 = _prepare(spec, A, center)
    B32 = A32 if B is None else _prepare(spec, B, center)
    n_a, n_b = A32.shape[0], B32.shape[0]
    out = np.empty((n_a, n_b), dtype=NUMPY_FLOAT)
    if n_a == 0 or n_b == 0:
        return out

    device = inference_device(backend)
    rows = _block_rows(n_a, n_b, device, block_rows)
    needs_norms = spec.kernel == "rbf"

    with use_device(device):
        transform = _tile_transform(mx, spec, device)
        right = mx.array(B32, dtype=mx.float32)
        right_t = right.T
        b_sq = mx.sum(right * right, axis=1) if needs_norms else None
        for start in range(0, n_a, rows):
            end = min(start + rows, n_a)
            block = mx.array(A32[start:end], dtype=mx.float32)
            a_sq = mx.sum(block * block, axis=1) if needs_norms else None
            tile = transform(block @ right_t, a_sq, b_sq)
            mx.eval(tile)
            out[start:end] = np.asarray(tile)
    return out


def kernel_combine(
    spec: KernelSpec,
    A: np.ndarray,
    B: np.ndarray,
    weights: np.ndarray,
    *,
    backend: str = "cpu",
    center: np.ndarray | None = None,
    block_rows: int | None = None,
) -> np.ndarray:
    """``K(A, B) @ weights``, blocked over ``A``, returned as float64.

    Parameters
    ----------
    spec : KernelSpec
    A : ndarray of shape (n_a, n_features)
    B : ndarray of shape (n_b, n_features)
    weights : ndarray of shape (n_b, n_outputs)
    backend : {"mlx", "cpu"}, default="cpu"
    center : ndarray of shape (n_features,), optional
    block_rows : int, optional

    Returns
    -------
    ndarray of shape (n_a, n_outputs), float64

    Notes
    -----
    The kernel entries are computed in float32 on device — that is where the
    arithmetic is, ``n_a · n_b · n_features`` — and the contraction against
    ``weights`` is done in float64 on the host, because doing it in float32 would
    add an accumulation error over every support vector on top of the kernel's own
    rounding. That precision is not free: with many one-vs-one pairs the host
    contraction measured 39–47% of ``predict``. It is therefore overlapped with
    the device, which computes the next tile while the host contracts this one,
    and done a chunk of rows at a time rather than through a float64 copy of the
    whole tile.

    Each row of ``A`` is reduced on its own, so ``block_rows`` changes the result
    only at the level of rounding — a BLAS may choose a different kernel for a
    block of a handful of rows — and, for a given input and blocking, the output
    is deterministic.
    """
    import mlx.core as mx

    A32 = _prepare(spec, A, center)
    B32 = _prepare(spec, B, center)
    n_a, n_b = A32.shape[0], B32.shape[0]
    W = np.asarray(weights, dtype=np.float64)
    out = np.zeros((n_a, W.shape[1]), dtype=np.float64)
    if n_a == 0 or n_b == 0:
        return out

    device = inference_device(backend)
    with use_device(device):
        right = mx.array(B32, dtype=mx.float32)
        b_sq = mx.sum(right * right, axis=1) if spec.kernel == "rbf" else None
        _combine_tiles(mx, spec, device, A32, right, b_sq, W, out, block_rows)
    return out


#: See :attr:`KernelRows.prefetch_rows`. 16–64 measured alike at 8 000 × 128;
#: much more fills the cache with rows fetched before the ranking meant anything.
_GPU_PREFETCH_ROWS = 32


@dataclass(slots=True)
class KernelRows:
    """Device-resident training data answering single-row kernel queries.

    Owned by whoever built it and released with it. Deliberately not a
    module-level cache keyed on the identity of the caller's array: an SMO fit
    that silently reused another fit's device tensors would produce a wrong model
    that no test distinguishes from a right one.

    Attributes
    ----------
    X : ndarray of shape (n_samples, n_features)
        Float32 C-ordered copy, already translated when ``spec.uses_center``.
    spec : KernelSpec
    backend : {"mlx", "cpu"}

    Notes
    -----
    Rows are computed on demand, never all ``n`` of them: a decomposition solver
    touches a scattered subset, and on-demand rows are the only shape of this
    problem that fits in bounded memory at large ``n``.

    What a row costs on the GPU is almost entirely the round trip — one
    synchronous evaluation, about 150–250 µs whether the row is ``n = 2 000`` or
    ``20 000`` long — so :meth:`rows` exists to pay it once for many rows: ``k``
    rows are one ``(n, d) @ (d, k)`` product, measured at 8–9 µs per row for
    ``k = 32`` against 200–250 µs for one at a time.

    :meth:`rows` evaluates every row, a single one included, through that product
    and the same tile transform :func:`kernel_matrix` uses; a lone row is a
    two-column product with itself repeated. A one-column product would be a
    matrix-*vector* product, which the backends reduce in a different order, so a
    row's value would depend on how many other rows were fetched alongside it — and
    a solver prefetching rows would take a different path from one that did not.
    As it is, on the GPU a row's value does not depend on its batch, and equals
    the corresponding row of :func:`kernel_matrix`'s Gram (measured on MLX 0.32
    over every shape tried; neither is a documented property of MLX, so neither
    is promised). On the CPU device it can differ in the last bit: the vectorized
    ``exp``/``tanh`` finish a tile with a scalar tail whose rounding differs, and
    where an entry falls in the tile depends on the batch. What is guaranteed on
    both is that a given request returns the same bits every time.

    :meth:`row` is the one-at-a-time path, and it differs by device. On the GPU it
    is ``rows([index])[0]``, at no extra cost. On the CPU device the per-call
    overhead is small, a batch buys little (13–22 µs per row, against 40–75 µs for
    a matrix-vector product), and the two-column product costs twice the
    matrix-vector one — so there :meth:`row` *is* the matrix-vector product, and
    its last bits can differ from :meth:`rows`'. A solver should therefore draw
    the rows it iterates on from one of the two, on the CPU device; the SVC fit
    uses :meth:`rows` there only to rebuild a gradient in bulk.
    """

    X: np.ndarray
    spec: KernelSpec
    backend: str
    _device_X: object = None  # mx.array (n, d)
    _device_sq: object = None  # mx.array (n,), only for rbf
    _tile: object = None  # f(dot, a_sq, b_sq), from _tile_transform
    _device_kind: str = "cpu"  # the MLX device ``backend`` resolves to here

    def __post_init__(self) -> None:
        import mlx.core as mx

        self._device_kind = inference_device(self.backend)
        with use_device(self._device_kind):
            data = mx.array(self.X, dtype=mx.float32)
            self._device_X = data
            self._tile = _tile_transform(mx, self.spec, self._device_kind)
            if self.spec.kernel == "rbf":
                self._device_sq = mx.sum(data * data, axis=1)
                mx.eval(data, self._device_sq)
            else:
                mx.eval(data)

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def prefetch_rows(self) -> int:
        """Further rows worth fetching alongside a cache miss, on this device.

        ``32`` on the GPU, where a call costs a round trip however many rows it
        carries: prefetching the near-violators this way measured 1.5–2.4× on rbf
        SVC fits of 8 000–16 000 rows (1.0× at 32 000, where the solver's own
        per-iteration arithmetic dominates instead). ``0`` on the CPU device, where a batched row
        costs a third of a lone one rather than a twentieth: prefetching 32 rows
        measured 0.82–0.87× there, and 8 rows 1.02–1.03×, which does not buy
        the batch-dependent last bits that mixing :meth:`row` and :meth:`rows`
        brings on that device (see Notes).
        """
        return _GPU_PREFETCH_ROWS if self._device_kind == "gpu" else 0

    def row(self, index: int) -> np.ndarray:
        """``K(X[index], X)`` as a float32 array of length ``n_samples``."""
        if self._device_kind == "gpu":
            return self.rows(np.array([index]))[0]

        import mlx.core as mx

        index = int(index)
        if not 0 <= index < self.n_samples:
            raise IndexError(f"row index must lie in [0, {self.n_samples}); got {index}")
        with use_device(self._device_kind):
            data = self._device_X
            dot = (data @ data[index][:, None])[:, 0]
            if self.spec.kernel == "linear":
                out = dot
            elif self.spec.kernel == "rbf":
                sq = self._device_sq
                squared = sq[index] + sq - 2.0 * dot
                out = mx.exp(-self.spec.gamma * mx.maximum(squared, 0.0))
            else:
                z = self.spec.gamma * dot + self.spec.coef0
                if self.spec.kernel == "sigmoid":
                    out = mx.tanh(z)
                else:
                    out = _integer_power(mx, z, int(self.spec.degree))
            mx.eval(out)
            return np.asarray(out, dtype=NUMPY_FLOAT)

    def rows(self, indices) -> np.ndarray:
        """``K(X[indices], X)`` as a float32 array of shape ``(len(indices), n_samples)``.

        One device round trip however many rows are asked for. The result owns
        its memory, so a caller may keep some rows and drop others without the
        ones it keeps pinning a buffer shared with the ones it dropped.
        """
        import mlx.core as mx

        index = np.asarray(indices).reshape(-1)
        count = int(index.shape[0])
        n = self.n_samples
        if count == 0:
            return np.empty((0, n), dtype=NUMPY_FLOAT)
        if not np.issubdtype(index.dtype, np.integer):
            raise TypeError(f"row indices must be integers, got dtype {index.dtype}")
        # An MLX gather does not bounds-check: an index past the end reads
        # whatever is there and returns it as a kernel row.
        if int(index.min()) < 0 or int(index.max()) >= n:
            raise IndexError(f"row indices must lie in [0, {n}); got {index.min()}..{index.max()}")
        # See the class Notes: one row is a two-column product, never a GEMV.
        chosen = np.repeat(index, 2) if count == 1 else index

        with use_device(self._device_kind):
            data = self._device_X
            selected = mx.array(chosen.astype(np.int32))
            a_sq = b_sq = None
            if self.spec.kernel == "rbf":
                b_sq = self._device_sq
                a_sq = b_sq[selected]
            # (n, d) @ (d, k) rather than (k, d) @ (d, n): the same entries, but
            # on the CPU device the latter's reduction order changed with k for
            # small shapes, and batch-independence is the point of this method.
            dot = (data @ data[selected].T).T
            tile = self._tile(dot, a_sq, b_sq)
            mx.eval(tile)
            return np.array(np.asarray(tile)[:count], dtype=NUMPY_FLOAT, order="C")


def build_kernel_rows(
    X: np.ndarray,
    spec: KernelSpec,
    *,
    backend: str,
    center: np.ndarray | None = None,
) -> KernelRows:
    """Build a :class:`KernelRows` over a float32 copy of ``X``.

    The copy is deliberate, and forced rather than incidental: ``_prepare`` returns
    its argument unchanged when it is already float32 and C-ordered, so without the
    explicit copy a row source could alias the caller's buffer and change its own
    answers when the caller mutated their array, mid-fit.
    """
    prepared = _prepare(spec, X, center)
    if prepared is X or np.shares_memory(prepared, X):
        prepared = prepared.copy()
    return KernelRows(X=prepared, spec=spec, backend=backend)
