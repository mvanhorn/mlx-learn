"""Blocked SVM kernel evaluation in MLX.

Four kernels — ``linear``, ``poly``, ``rbf``, ``sigmoid`` — with scikit-learn's
``gamma`` resolution, evaluated one row block at a time so that the working set on
the device is bounded by a byte budget rather than by ``n_samples²``. Nothing here
knows what an estimator is; everything takes and returns arrays and frozen
plain-data objects.

Three things this module is responsible for.

**Never requiring a full Gram.** :class:`KernelRows` answers "give me row ``i``"
from device-resident training data, and :func:`kernel_combine` folds
``K(A, B) @ W`` block by block. The dense ``n × n`` matrix is materialized only
when a caller explicitly asks for it via :func:`kernel_matrix`, which is what an
SMO fit does when — and only when — its row-cache budget already permits holding
every row.

**Exactness under float32.** The squared distance inside the RBF kernel comes from
the expanded identity ``‖a‖² + ‖b‖² − 2⟨a, b⟩``, which is hostile in float32 for
data far from the origin (``docs/fp32_policy.md`` §2.1). Distances are
translation-invariant, so both matrices are translated by a common ``center``
before the identity is applied — exact mathematics, and it is what keeps the
cancellation at the variance scale instead of the coordinate scale. Centering is
applied *only* to the RBF kernel: ``linear``, ``poly`` and ``sigmoid`` read the raw
inner product, which is not translation-invariant, and shifting it would compute a
different kernel.

**Integer powers by repeated squaring.** ``poly`` is ``(γ⟨a,b⟩ + coef0)ᵈ`` with an
integer degree. Evaluating that as a floating-point power returns NaN whenever the
base is negative, which happens routinely for ``coef0=0`` and a negative inner
product. LIBSVM uses binary exponentiation and so does this module, so a negative
base raised to an odd degree yields the negative value it should.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np

from .._common.arrays import NUMPY_FLOAT
from .._common.config import get_tuning
from .._common.device import use_device
from ..exceptions import UnsupportedParameterError

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


def resolve_gamma(gamma, X: np.ndarray) -> float:
    """Resolve ``gamma`` the way scikit-learn resolves it.

    Parameters
    ----------
    gamma : {"scale", "auto"} or float
    X : ndarray of shape (n_samples, n_features)
        The validated training matrix, float64. The variance is taken over *all*
        entries — ``X.var()``, not a per-feature variance — because that is what
        scikit-learn does and ``gamma`` has to mean the same number here.

    Returns
    -------
    float

    Notes
    -----
    ``"scale"`` is ``1 / (n_features * X.var())``, falling back to ``1.0`` when the
    variance is at or below ``1e-12``; a constant ``X`` would otherwise divide by
    zero. ``"auto"`` is ``1 / n_features``.

    The variance is computed on the float64 validated matrix, not on the float32
    copy mlxlearn computes with. A float32 variance differs in the ninth
    significant digit, and that difference would propagate into every kernel entry
    of every fit — a systematic offset from scikit-learn bought for nothing.

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
            return 1.0 / (X.shape[1] * variance) if variance > 1e-12 else 1.0
        if gamma == "auto":
            return 1.0 / X.shape[1]
        raise UnsupportedParameterError(
            f"gamma={gamma!r} is not a value mlxlearn resolves. Use 'scale', 'auto', "
            "or a positive float.",
            parameter="gamma",
            value=gamma,
        )
    value = float(gamma)
    if value <= 0.0:
        # scikit-learn's constraint is Interval(Real, 0.0, None, closed="left"),
        # so gamma=0.0 is legal there and produces a degenerate but well-defined
        # model. mlxlearn's kernels do not implement it — and because
        # scikit-learn *can* serve the request, this must be a CapabilityError so
        # that patched mode falls back instead of raising. A plain ValueError here
        # propagated straight through mlx_guard and broke Layer 2 too.
        raise UnsupportedParameterError(
            f"mlxlearn does not implement gamma={value!r}; the MLX kernels require a "
            "strictly positive gamma. scikit-learn accepts it, so under "
            "patch_sklearn() this falls back rather than failing.",
            parameter="gamma",
            value=value,
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
# device evaluation
# --------------------------------------------------------------------------------------


def _device(backend: str) -> str:
    return "gpu" if backend == "mlx" else "cpu"


def _block_rows(n_rows: int, n_cols: int, backend: str, block_rows: int | None) -> int:
    """How many rows of the left operand to evaluate at once.

    Derived from the same byte budget the distance kernel uses, halved because a
    block needs both the inner-product tile and the transformed tile alive at the
    same time. A fixed constant would be wrong by an order of magnitude on both an
    8 GB laptop and a 128 GB desktop.
    """
    if block_rows is not None:
        return max(1, min(int(block_rows), max(n_rows, 1)))
    tuning = get_tuning()
    budget = (
        tuning.distance_block_bytes if backend == "mlx" else tuning.cpu_distance_block_bytes
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


def _tile_transform(mx, spec: KernelSpec, backend: str):
    """``f(dot, a_sq, b_sq)`` turning an inner-product tile into a kernel tile.

    The fused rbf kernel is used on the GPU only. On MLX's CPU device the same
    compiled graph measured 2.3–2.6× *slower* than the eager ops, and its first
    call per process shells out to a C compiler for about a second — neither of
    which the CPU path, which Layer 1 takes for every problem below the SVC
    crossover, should pay for an identical answer.
    """
    if spec.kernel == "rbf" and backend == "mlx":
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

    rows = _block_rows(n_a, n_b, backend, block_rows)
    needs_norms = spec.kernel == "rbf"

    with use_device(_device(backend)):
        transform = _tile_transform(mx, spec, backend)
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

    rows = _block_rows(n_a, n_b, backend, block_rows)
    needs_norms = spec.kernel == "rbf"

    with use_device(_device(backend)):
        right = mx.array(B32, dtype=mx.float32)
        b_sq = mx.sum(right * right, axis=1) if needs_norms else None

        def left(start, stop):
            block = mx.array(A32[start:stop], dtype=mx.float32)
            return block, (mx.sum(block * block, axis=1) if needs_norms else None)

        _combine_pipelined(
            mx, _tile_transform(mx, spec, backend), n_a, rows, left, right, b_sq, W, out
        )
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

    def __post_init__(self) -> None:
        import mlx.core as mx

        with use_device(_device(self.backend)):
            data = mx.array(self.X, dtype=mx.float32)
            self._device_X = data
            self._tile = _tile_transform(mx, self.spec, self.backend)
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
        carries: prefetching the near-violators this way measured 1.5–2.4× on SVC
        fits of 8 000–20 000 rows. ``0`` on the CPU device, where a batched row
        costs a third of a lone one rather than a twentieth: prefetching 32 rows
        measured 0.82–0.87× there, and 8 rows 1.02–1.03×, which does not buy
        the batch-dependent last bits that mixing :meth:`row` and :meth:`rows`
        brings on that device (see Notes).
        """
        return _GPU_PREFETCH_ROWS if self.backend == "mlx" else 0

    def row(self, index: int) -> np.ndarray:
        """``K(X[index], X)`` as a float32 array of length ``n_samples``."""
        if self.backend == "mlx":
            return self.rows(np.array([index]))[0]

        import mlx.core as mx

        index = int(index)
        if not 0 <= index < self.n_samples:
            raise IndexError(f"row index must lie in [0, {self.n_samples}); got {index}")
        with use_device(_device(self.backend)):
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

        with use_device(_device(self.backend)):
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
