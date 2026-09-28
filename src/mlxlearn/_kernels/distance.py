"""Blocked exact Euclidean distance and k-nearest-neighbor selection in MLX.

The whole computation stays on the device. The ancestor materialized each tile's
top-k back to the host and merged it in a Python loop, which meant the GPU did the
matmul and the interpreter did everything else; for a hundred thousand queries and
a hundred tiles that is on the order of 10⁸ interpreted iterations, and it is why
its measured KNN speedups ranged down to 0.4×. Here, the running top-k is an MLX
array, tiles are merged with a concatenate-and-repartition on the device, and the
only host transfer is the final ``(n_query, k)`` result — one copy per query block.

Three correctness properties this module is responsible for:

**Exactness under float32.** Candidates are *selected* with the expanded identity
``‖q‖² + ‖t‖² − 2⟨q,t⟩`` — one matmul, and numerically hostile — on training rows
translated by their mean and scaled by a power of two, and the survivors' distances
are then *refined* directly as ``Σ(t − q)²`` from the untranslated coordinates. A
fixed over-selection margin alone is not enough: on clustered or near-duplicate data
the Gram noise exceeds the gaps between neighbors and true neighbors are silently
dropped. So every row is **certified**. A rigorous float32 error bound on the
identity proves that no training row left out of the candidate set can beat the
k-th refined distance; a row it cannot certify is searched again with a much wider
candidate set, and if that fails too, exactly, by direct differences against every
training row. See ``docs/fp32_policy.md`` §2.1.

**A total order.** Results are the first ``k`` training rows under ``(refined
distance, training index)``, so exact ties break on the smaller index — including
ties at the k-th boundary, which the certificate treats as uncertain — and the
answer for ``k`` is a prefix of the answer for any larger ``k``. Across devices the
answers agree wherever the refined distances are exact (integer and binary data, for
example); the GPU and the CPU may round a long sum differently.

**Blocking invariance.** Any combination of query- and train-block sizes yields
identical output. The tests sweep this.
"""

from __future__ import annotations

import functools
import math
import threading
from dataclasses import InitVar, dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np

from .._common.arrays import INDEX_DTYPE, NUMPY_FLOAT
from .._common.config import get_tuning
from .._common.device import physical_memory_bytes, use_device
from ..exceptions import UnsupportedInputError

__all__ = ["NeighborIndex", "build_index", "resolve_blocks"]

#: Extra candidates carried per tile beyond ``k`` on the first pass — at least this
#: many, and more for wide data and large ``k`` (see :func:`_first_pass_carry`). This
#: is not what makes the answer exact — the certificate is — but it is what lets the
#: certificate succeed on the first pass for ordinary data.
_OVERSELECT = 4

#: Candidates carried on the second pass over rows the first pass could not certify:
#: ``max(4 * k_carry, k_carry + _WIDEN_EXTRA)``. When a cluster is tight relative to
#: its distance from the mean, all of its members sit inside the Gram noise band
#: together, so the second pass has to be able to carry a whole cluster —
#: ``make_blobs`` at ``cluster_std=0.01`` puts ~250 points in each.
_WIDEN_EXTRA = 256

#: Segment widths for two-stage selection on the GPU: at least 512, at least four
#: times the number kept, at most 8192. MLX's GPU ``argpartition`` is effectively a
#: full sort of the row, which made selection cost several times the Gram matmul on
#: a wide tile. Partitioning segments first and then the survivors measured 2.5-4×
#: faster keeping 14 of 100 000 and 1.2-2.5× keeping 64-1030, with identical sets.
_SELECT_SEGMENT = 512
_SELECT_SEGMENT_MAX = 8192

#: Peak device memory per byte of distance tile. The budget below is for the whole
#: working set, not the tile alone: the matmul output and the fused tile expression
#: each cost a tile, and a full-row GPU ``argpartition`` — still used when ``k`` is
#: too large for segmented selection — allocates five more. Measured with
#: ``mx.get_peak_memory``: the unfactored budget peaked at 7.0× (7.2 GB against
#: 1 GiB); with this factor, 0.2-0.6 GB. Query time was flat across factors 1-7. The
#: CPU evaluates the tile expression unfused, a tile or two more, which the GPU's
#: argpartition share covers there.
_TILE_WORKSET_FACTOR = 7

#: Peak device memory per byte of the ``(rows, k_carry, n_features)`` refinement
#: gather: the gathered rows, the fused difference, and — for float64 input — the
#: gathered residuals. Unbudgeted, ``k=1000`` at 768 features reached 7.4 GB.
_REFINE_WORKSET_FACTOR = 4

#: Queries on small-integer training data stay on the exact integer path when they
#: are multiples of ``2**-_DYADIC_SHIFTS`` — half-integers, quarters — because
#: scaling both sides by a power of two keeps every product an integer.
_DYADIC_SHIFTS = 4

#: float32 unit roundoff.
_U = 2.0**-24

#: float32's largest finite value.
_FLOAT32_MAX = float(np.finfo(np.float32).max)

#: Largest distance from the training mean mlxlearn will subtract in float32. Twice
#: this must stay finite, because refinement subtracts two such coordinates before
#: scaling them.
_SPAN_LIMIT = _FLOAT32_MAX / 8.0

#: Room, in bits, the scaling leaves between the training span and float32's
#: overflow for queries lying farther out: one about 2**20 times the span from the
#: training mean is still served. Every bit of it is a bit less for resolving small
#: gaps under a large span; see _span_exponent.
_QUERY_HEADROOM_BITS = 20

#: The smallest scaled squared distance refinement is trusted with. Below it, the
#: squares of per-coordinate differences may have left float32's normal range — the
#: GPU flushes those to zero and the CPU keeps them as subnormals, so the devices
#: disagree and neither is right. At or above it, all that underflow can remove,
#: d·2**-126, is 2**-30·d of the value: far inside refinement's own rounding.
_RESOLVED_SQ = 2.0**-96

#: At fit, a training set whose own rows' nearest neighbors come within this factor
#: of _RESOLVED_SQ is refused, so that such data falls back at fit under
#: patch_sklearn() rather than failing its queries at inference.
_RESOLVED_MARGIN = 2.0**20

#: Training rows searched at fit for that check, spread evenly over the rows.
_PROBE_ROWS = 64

#: The exact search packs a training index into 32 bits of its sort key.
_MAX_TRAIN_ROWS = 2**32

#: Training matrices up to this many elements have their column extremes found on
#: the host at build: faster than the device round trip that reduction otherwise
#: costs (0.1 ms on the GPU), up to about 8000 x 32, where the two measured even.
_HOST_REDUCE_ELEMENTS = 1 << 18


@dataclass(frozen=True, slots=True)
class BlockPlan:
    """How a query is split into tiles."""

    query_rows: int
    train_cols: int

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"query_rows={self.query_rows}, train_cols={self.train_cols}"


def _distance_budget_bytes(backend: str) -> int:
    """Bytes the query working set may occupy on the device.

    Derived from installed memory and clamped, because a fixed constant is wrong by
    an order of magnitude on both an 8 GB Air and a 128 GB Studio. It bounds the
    *working set* — tile, selection temporaries and the refinement gather together —
    not merely the distance tile: bounding only the tile let the real peak reach
    seven times the budget, and MLX's buffer cache then kept that peak resident after
    the call returned. Wider tiles bought nothing once selection was segmented.
    """
    tuning = get_tuning()
    if backend != "mlx":
        return tuning.cpu_distance_block_bytes
    total = physical_memory_bytes()
    if total is None:
        return tuning.distance_block_bytes
    return int(
        min(max(total // 32, tuning.distance_block_bytes), tuning.distance_block_bytes_max)
    )


def _align_down(value: int, align: int) -> int:
    return (value // align) * align if align > 1 and value >= align else value


def resolve_blocks(
    n_query: int,
    n_train: int,
    *,
    backend: str,
    k_search: int,
    query_block_size: int | None = None,
    train_block_size: int | None = None,
) -> BlockPlan:
    """Choose tile sizes so the distance tile's working set fits the budget.

    Unlike the ancestor, the device kind is passed in rather than inferred from
    whether the byte budget happens to exceed a GPU-ish constant — conflating a
    memory budget with a device kind meant an explicit CPU budget above 256 MiB
    silently switched on GPU alignment, and the CPU path could never shrink its
    block no matter how wide the training set was.

    The refinement gather, which grows with ``k * n_features`` rather than with the
    training set, is budgeted separately, by row chunks inside a block, so that a
    large ``k`` does not also shrink the matmul.
    """
    if n_query <= 0 or n_train <= 0:
        return BlockPlan(query_rows=0, train_cols=0)

    tuning = get_tuning()
    budget = _distance_budget_bytes(backend)
    tile_budget = max(budget // _TILE_WORKSET_FACTOR, 1)
    itemsize = 4

    if query_block_size is not None:
        rows = max(1, min(int(query_block_size), n_query))
    else:
        rows = tuning.query_block_rows if backend == "mlx" else tuning.cpu_query_block_rows
        # Keep a full-width tile inside the budget when the training set is small
        # enough to be scanned in one pass; otherwise the train tiling handles it.
        full_width_rows = tile_budget // max(n_train * itemsize, 1)
        if 0 < full_width_rows < rows:
            align = tuning.query_block_align if backend == "mlx" else 1
            rows = _align_down(full_width_rows, align)
        rows = max(1, min(n_query, rows))

    min_cols = min(k_search, n_train)
    if train_block_size is not None:
        cols = max(min_cols, min(int(train_block_size), n_train))
    else:
        cols = tile_budget // max(rows * itemsize, 1)
        cols = max(min_cols, min(n_train, int(cols)))

    return BlockPlan(query_rows=int(rows), train_cols=int(cols))


def _compute_device(backend: str, device: str | None = None) -> str:
    """The MLX device an index built for ``backend`` runs on: ``device``, or else its own.

    ``"cpu"`` is MLX's CPU and ``"mlx"`` the GPU. A process with none — a model
    fitted on a Mac, unpickled somewhere without Metal — runs the identical float32
    code on MLX's CPU instead, unless the configuration names the GPU. That is the
    estimator's decision (:func:`~mlxlearn._common.device.inference_device`), made
    where the configuration may be read and passed in as ``device``: the kernels
    read no configuration, and the device a query runs on is the one diagnostics
    report.
    """
    if device is not None:
        return device
    return "gpu" if backend == "mlx" else "cpu"


@functools.lru_cache(maxsize=2)
def _kernels(device: str) -> SimpleNamespace:
    """The elementwise pieces for ``device``: fused on the GPU, eager on the CPU.

    On the GPU ``mx.compile`` fuses each expression into one kernel, so the tile
    costs one temporary where the unfused expression cost four, and the refinement
    difference one where it cost three. On MLX's CPU device the compiled graphs ran
    no faster, and each one shells out to a C compiler once per MLX version and
    ``$TMPDIR``: a 200-row fit and predict took 3-4 s on a fresh machine, against
    0.1 s eager. So the CPU evaluates the same expressions eagerly, as
    ``kernels._tile_transform`` does for the rbf tile; the answers are identical.
    Compiled functions are cached by MLX on input shape and stream only — never on
    data — and are built lazily so that importing mlxlearn does not import MLX.
    """
    import mlx.core as mx

    def center_scale(hi, center, scale):
        return (hi - center) * scale

    def center_scale_lo(hi, lo, center, center_lo, scale):
        return ((hi - center) + (lo - center_lo)) * scale

    def tile(q_sq, t_sq, gram):
        # The clamp only ever moves a value toward the truth: distances are >= 0.
        return mx.maximum((q_sq + t_sq) - 2.0 * gram, 0.0)

    def pair_sq(t, q, scale):
        diff = (t - q) * scale
        return diff * diff

    def pair_sq_lo(t, t_lo, q, q_lo, scale):
        diff = ((t - q) + (t_lo - q_lo)) * scale
        return diff * diff

    fns = {
        "center_scale": center_scale,
        "center_scale_lo": center_scale_lo,
        "tile": tile,
        "pair_sq": pair_sq,
        "pair_sq_lo": pair_sq_lo,
    }
    if device == "gpu":
        fns = {name: mx.compile(fn) for name, fn in fns.items()}
    return SimpleNamespace(**fns)


def _translate(fns, hi, lo, center, center_lo, scale, zero):
    """``(hi + lo − center − center_lo) · scale`` in float32, into the index's frame.

    ``lo`` is a float64 row's residual and ``center_lo`` a float64 center's, either
    of them ``None``. Every row, training or query, residual or not, is translated
    here, so all of them share one frame; with neither residual it is the plain
    ``(hi − center) · scale`` it always was.
    """
    if lo is None and center_lo is None:
        return fns.center_scale(hi, center, scale)
    lo = zero if lo is None else lo
    center_lo = zero if center_lo is None else center_lo
    return fns.center_scale_lo(hi, lo, center, center_lo, scale)


def _gamma(n: int) -> float:
    """Higham's γₙ = n·u / (1 − n·u): the relative error bound of an n-term float32 sum."""
    nu = n * _U
    return nu / (1.0 - nu) if nu < 0.5 else math.inf


def _scaled_limit(n_features: int) -> float:
    """Largest scaled coordinate whose Gram tile — ``4·d`` squares — stays finite."""
    return math.sqrt(_FLOAT32_MAX / (4.0 * max(n_features, 1)))


def _span_exponent(spread: float, n_features: int) -> int:
    """The index scales by ``2**-exponent``: exact, and only as far as float32 needs.

    A span below 1 is scaled up into [0.5, 1): data around 1e-20 squares into the
    subnormal range, which the GPU flushes to zero. A span whose squares, with
    ``2**_QUERY_HEADROOM_BITS`` of room for queries beyond it, would overflow is
    scaled down just below that, and no further. Anything between is left alone.

    Scaling down shrinks the gaps between neighbors along with the span. Every span
    used to be normalized into [0.5, 1), and one 1e20 fill value among unit-scale
    rows then put their squared gaps near 2**-134: the GPU returned every distance
    as zero and the neighbors in index order, with nothing raised. What even the
    smallest scale-down cannot resolve is refused; see :func:`_require_resolvable`.
    """
    if spread == 0.0:
        return 0
    exponent = math.frexp(spread)[1]
    if exponent <= 0:
        return max(exponent, -126)
    top = math.floor(math.log2(_scaled_limit(n_features))) - _QUERY_HEADROOM_BITS
    return max(0, exponent - top)


def _first_pass_carry(k_search: int, n_features: int, n_train: int) -> int:
    """Candidates to carry on the first pass.

    The certificate's noise band is proportional to ``γ_d``, and the number of
    neighbors crowding into it near the k-th distance grows with ``d`` and ``k``: on
    Gaussian data the rows that failed a first pass with a fixed margin of 4 went
    from none at d ≤ 64 to 29 of 200 at d=768, k=100 and all 200 at k=1000, and the
    margin that brought each back to zero tracked ``d²·u·k``. A second pass costs a
    whole extra search, so erring wide here is the cheaper mistake; the carry never
    more than doubles ``k``.
    """
    band = math.ceil(4.0 * n_features * n_features * _U * k_search)
    margin = max(_OVERSELECT, min(k_search, band))
    return min(n_train, k_search + margin)


@dataclass(frozen=True, slots=True)
class _DeviceState:
    """Everything the device holds for one index. Built whole, published once.

    Immutable, and assigned to the index only after every array in it has been
    evaluated, so a thread that sees it can use it — see ``_ensure_device``.
    """

    raw: Any  # (n, d) float32, uncentered; the index's ``X`` is a view of it
    raw_lo: Any  # (n, d) float32 residual of float64 input, or None
    sel: Any  # (n, d) (X - center) * 2**-exponent: the Gram operand
    sel_t: Any  # sel.T
    sel_sq: Any  # (n,) squared row norms of sel
    center: Any  # (d,) float32
    center_lo: Any  # (d,) float32 residual of a float64 center, or None
    scale: Any  # 0-d, 2**-exponent
    zero: Any  # 0-d, 0.0
    exponent: int
    spread: float  # largest |X - center|, unscaled
    norms_sorted: np.ndarray  # ascending float64 row norms of sel, for the certificate
    raw_norm_bound: float  # upper bound on any training row's ||x||, unscaled
    int_limit: int  # > 0 when the rows are small integers and the Gram is exact
    int_max: float  # largest |x| in the training rows, integer path only
    raw_t: Any  # raw.T, integer path only
    raw_sq: Any  # (n,) exact squared norms of raw, integer path only
    index_dtype: Any


@dataclass(slots=True)
class _QueryBlock:
    """One block of query rows, in every representation the search needs."""

    hi: Any  # (B, d) float32 query rows
    lo: Any  # (B, d) float32 residual of float64 queries, or None
    sel: Any  # (B, d) translated and scaled like the training rows
    sq: Any  # (B, 1) squared norms of sel
    raw_norm: np.ndarray | None  # (B,) ||q||, only when a residual is in play
    int_shift: int | None  # 2**shift makes both sides small integers; None if not

    @property
    def n(self) -> int:
        return int(self.hi.shape[0])

    def take(self, mx, rows: np.ndarray) -> _QueryBlock:
        index = mx.array(rows.astype(np.int32))

        def pick(a):
            return None if a is None else mx.take(a, index, axis=0)

        return _QueryBlock(
            hi=pick(self.hi),
            lo=pick(self.lo),
            sel=pick(self.sel),
            sq=pick(self.sq),
            raw_norm=None if self.raw_norm is None else self.raw_norm[rows],
            int_shift=self.int_shift,
        )


@dataclass(slots=True, eq=False)
class NeighborIndex:
    """Device-resident training data for repeated neighbor queries.

    Owned by the fitted estimator and released with it. Deliberately *not* a
    module-level cache keyed on the identity of a user's array: the ancestor did
    that and it produced stale results whenever a caller mutated an array in
    place, leaked a full float32 copy plus its transpose for the lifetime of the
    process, and was not thread-safe. An agent refitting a probe on new data must
    never get the previous fit's coefficients back.

    Attributes
    ----------
    X : ndarray of shape (n_samples, n_features)
        The training rows in float32, C-ordered, and never the caller's buffer: an
        estimator that aliases the caller's array changes its own predictions when
        the caller mutates it. Once the device copy exists this is a read-only,
        zero-copy view of it, so on unified memory the host and the device share
        one buffer instead of holding two.
    center : ndarray of shape (n_features,)
        The translation applied before the Gram identity: the training mean.
        Distances are translation-invariant, so any shift is exact in real
        arithmetic; the mean is the one that keeps float32 cancellation manageable.
        float64 for wider-than-float32 input, and applied as a float32 value and
        residual, like ``X``: rounded to float32, a mean near 1.7e9 is up to 64 away
        from the true one, and that offset alone swamped the certificate for
        timestamps a few seconds apart. float32 — the mean, rounded — for float32
        input.
    backend : {"mlx", "cpu"}
        Which MLX device the index was built for.
    device : {"gpu", "cpu"}, optional
        Init only: the MLX device to build on, as the caller resolved it; by
        default the one ``backend`` names. Not kept, and not pickled: each query
        says where it runs.
    X_lo : ndarray of shape (n_samples, n_features) or None
        For wider-than-float32 input, the float32 residual ``X64 − float32(X64)``.
        Refinement and centering use ``X + X_lo``, so a float64 caller's precision
        is not thrown away by the cast — a column of Unix timestamps keeps its
        seconds. ``None`` when the input was float32 or exactly representable.
    """

    X: np.ndarray
    center: np.ndarray
    backend: str
    X_lo: np.ndarray | None = None
    _state: _DeviceState | None = field(default=None, repr=False)
    _lock: Any = field(default_factory=threading.Lock, repr=False)
    device: InitVar[str | None] = None

    # ------------------------------------------------------------------ build

    def __post_init__(self, device: str | None) -> None:
        self._ensure_device(device)

    def _ensure_device(self, device: str | None = None) -> _DeviceState:
        """Build and publish the device state. Idempotent and thread-safe.

        It doubles as the rebuild path after unpickling, and there the first query
        may well be a burst from a thread pool. This method used to publish the
        device arrays one at a time, still unevaluated, so a second thread could pass
        the "already built" check and read ``None`` — or evaluate a lazy graph
        belonging to another thread's stream, which MLX answers with "There is no
        Stream(gpu, N) in current thread" or a segfault. Here the state is built
        into locals, evaluated, and published with one assignment, under
        double-checked locking so that concurrent first queries also do not each
        upload their own copy of the training set.
        """
        state = self._state
        if state is not None:
            return state
        with self._lock:
            if self._state is None:
                state, x_view, lo_view = _build_state(
                    self.X, self.X_lo, self.center, _compute_device(self.backend, device)
                )
                self.X = x_view
                self.X_lo = lo_view
                self._state = state  # published last: the unlocked check reads only this
            return self._state

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    # ------------------------------------------------------------------ pickle

    def __getstate__(self) -> dict:
        """Pickle the host data only; the device tensors are derived.

        Keeping them would multiply the pickle — the translated matrix, its row
        norms and their sorted copy are all recomputable from ``X``, ``X_lo`` and
        ``center`` — and would embed device buffers in a file that may be loaded on
        a machine with a different MLX build or no GPU at all.
        """
        state = {"X": self.X, "center": self.center, "backend": self.backend}
        if self.X_lo is not None:
            state["X_lo"] = self.X_lo
        return state

    def __setstate__(self, state: dict) -> None:
        self.X = state["X"]
        self.center = state["center"]
        self.backend = state["backend"]
        self.X_lo = state.get("X_lo")
        # A lock is process-local and never pickled; each unpickled index gets its own.
        self._lock = threading.Lock()
        self._state = None
        # Rebuilt lazily on the first query rather than here, so that unpickling
        # never touches a device, and on the device that query runs on; see
        # _compute_device.

    # ------------------------------------------------------------------ query

    def query(
        self,
        Q: np.ndarray,
        k: int,
        *,
        return_distance: bool = True,
        exclude_self: bool = False,
        query_block_size: int | None = None,
        train_block_size: int | None = None,
        stats: dict | None = None,
        device: str | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray]:
        """Return the ``k`` nearest training rows for each row of ``Q``.

        Parameters
        ----------
        Q : ndarray of shape (n_queries, n_features)
            Query rows, float32 or float64. float64 rows are split into a float32
            value and residual exactly as the training rows are, so their precision
            is used rather than discarded by a cast.
        k : int
            Neighbors to return, before ``exclude_self`` removes one.
        return_distance : bool, default=True
            Whether to compute and return Euclidean distances.
        exclude_self : bool, default=False
            Treat ``Q`` as the training matrix itself and drop each row's own
            index. Requires ``len(Q) == n_samples``. Passing this index's own ``X``
            queries with the device-resident training rows directly.
        stats : dict, optional
            If given, filled with how the rows were answered: ``"rows"``,
            ``"widened_rows"`` (the first pass could not certify them) and
            ``"exact_rows"`` (nor could a wider one, so they were searched exactly).
            The kernel knows nothing of estimators, so it is the caller's to record.
        device : {"gpu", "cpu"}, optional
            The MLX device to search on, as the caller resolved it; by default the
            one the index was built for (``backend``).

        Returns
        -------
        distances : ndarray of shape (n_queries, k) or None
            Euclidean (not squared) distances, float64, ascending.
        indices : ndarray of shape (n_queries, k)
            Training-row indices, int64: the first ``k`` under ``(distance, index)``.

        Notes
        -----
        ``exclude_self`` compares against each row's **global** index. The
        ancestor compared against the block-local row number, so every query block
        after the first tested the wrong identity and a row could be returned as
        its own nearest neighbor — verified on 1500 rows with default blocking.
        """
        import mlx.core as mx

        # Decided before the device build, which swaps ``X`` for a view of the
        # device copy: the caller handed us the old object.
        self_query = exclude_self and Q is self.X
        device = _compute_device(self.backend, device)
        state = self._ensure_device(device)
        n_query = int(Q.shape[0])
        n_train = self.n_samples
        k = int(k)

        if exclude_self and n_query != n_train:
            raise ValueError(
                "exclude_self requires the query matrix to be the training matrix: "
                f"got {n_query} query rows and {n_train} training rows."
            )
        k_search = k + int(exclude_self)
        if k_search > n_train:
            limit = n_train - int(exclude_self)
            raise ValueError(f"n_neighbors must be <= {limit}, got {k}.")

        if n_query == 0:
            empty_idx = np.empty((0, k), dtype=INDEX_DTYPE)
            empty_dist = np.empty((0, k), dtype=np.float64) if return_distance else None
            return empty_dist, empty_idx

        if not self_query:
            Q = np.asarray(Q)
            if Q.ndim != 2 or Q.shape[1] != self.n_features:
                raise ValueError(
                    f"Q must have shape (n_queries, {self.n_features}); got {Q.shape}."
                )

        # Carry extra candidates so the certificate usually succeeds first time.
        k_carry = _first_pass_carry(k_search, self.n_features, n_train)
        # Budgeted for the device the query runs on, which for a GPU-fitted index
        # loaded on a GPU-less host is MLX's CPU, not the device it was built for.
        plan_backend = "mlx" if device == "gpu" else "cpu"
        plan = resolve_blocks(
            n_query,
            n_train,
            backend=plan_backend,
            k_search=k_carry,
            query_block_size=query_block_size,
            train_block_size=train_block_size,
        )

        out_idx = np.empty((n_query, k), dtype=INDEX_DTYPE)
        out_dist = np.empty((n_query, k), dtype=np.float64) if return_distance else None

        search = _Search(
            index=self,
            state=state,
            mx=mx,
            k_search=k_search,
            k_carry=k_carry,
            segmented=device == "gpu",
            fns=_kernels(device),
            plan_backend=plan_backend,
            train_block_size=train_block_size,
            counts={"rows": n_query, "widened_rows": 0, "exact_rows": 0},
        )
        with use_device(device):
            for start in range(0, n_query, plan.query_rows):
                end = min(start + plan.query_rows, n_query)
                block = search.query_block(Q, start, end, self_query=self_query)
                idx_np, sq_np = search.block(block, plan.train_cols)

                if exclude_self:
                    idx_np, sq_np = _drop_self(idx_np, sq_np, offset=start)

                out_idx[start:end] = idx_np[:, :k]
                if return_distance:
                    out_dist[start:end] = np.sqrt(sq_np[:, :k])

        if stats is not None:
            stats.update(search.counts)
        return out_dist, out_idx


@dataclass(slots=True)
class _Search:
    """One ``query`` call's worth of search machinery over a built index."""

    index: NeighborIndex
    state: _DeviceState
    mx: Any
    k_search: int
    k_carry: int
    segmented: bool
    fns: SimpleNamespace  # _kernels() for the device the search runs on
    plan_backend: str  # "mlx" or "cpu": whose memory budget the blocks follow
    train_block_size: int | None
    counts: dict

    # ------------------------------------------------------------- query rows

    def query_block(self, Q, start: int, end: int, *, self_query: bool) -> _QueryBlock:
        """Prepare rows ``start:end`` of the query in every representation needed."""
        mx, state = self.mx, self.state
        fns = self.fns

        if self_query:
            # The training rows themselves: already on the device, already
            # translated, and identical bit for bit to what they are compared with.
            lo = None if state.raw_lo is None else state.raw_lo[start:end]
            raw_norm = None
            if lo is not None:
                host = np.asarray(self.index.X[start:end], dtype=np.float64)
                raw_norm = np.sqrt(np.einsum("ij,ij->i", host, host))
            return _QueryBlock(
                hi=state.raw[start:end],
                lo=lo,
                sel=state.sel[start:end],
                sq=state.sel_sq[start:end, None],
                raw_norm=raw_norm,
                int_shift=0 if state.int_limit > 0 else None,
            )

        rows = Q[start:end]
        hi_np = np.ascontiguousarray(rows, dtype=NUMPY_FLOAT)
        wide = rows.dtype != NUMPY_FLOAT
        rows64 = np.asarray(rows, dtype=np.float64)
        lo_np = None
        if wide:
            lo_np = (rows64 - hi_np).astype(NUMPY_FLOAT)
            if not lo_np.any():
                lo_np = None

        # A query far enough from the training data overflows float32 even after
        # scaling. Inference cannot fall back — the backend is sticky — so this is
        # an error rather than a row of infinities.
        offset = float(np.max(np.abs(rows64 - self.index.center.astype(np.float64))))
        if not math.isfinite(offset) or offset > _SPAN_LIMIT or (
            math.ldexp(offset, -state.exponent) > _scaled_limit(self.index.n_features)
        ):
            raise UnsupportedInputError(
                f"A query row lies {offset:.3g} from the training data's mean, which "
                "overflows float32 distance arithmetic at the training data's scale. "
                "Rescale the queries, or compute these neighbors with scikit-learn.",
                reason="float32-overflow",
            )

        hi = mx.array(hi_np)
        lo = None if lo_np is None else mx.array(lo_np)
        sel = _translate(fns, hi, lo, state.center, state.center_lo, state.scale, state.zero)
        sq = mx.sum(sel * sel, axis=1, keepdims=True)

        raw_norm = None
        if lo is not None or state.raw_lo is not None:
            raw_norm = np.sqrt(np.einsum("ij,ij->i", rows64, rows64))

        int_shift = None
        if state.int_limit > 0 and lo_np is None:
            int_shift = _dyadic_shift(hi_np, state.int_max, state.int_limit)
        return _QueryBlock(hi=hi, lo=lo, sel=sel, sq=sq, raw_norm=raw_norm, int_shift=int_shift)

    # ------------------------------------------------------------------ search

    def block(self, qb: _QueryBlock, train_cols: int) -> tuple[np.ndarray, np.ndarray]:
        """The first ``k_search`` neighbors of each row of ``qb``.

        Returns host arrays: indices, and float64 squared distances in the data's
        own units, ordered by ``(distance, index)``.
        """
        mx, state = self.mx, self.state
        n_train = self.index.n_samples

        if qb.int_shift is not None:
            # Small integers: the Gram identity on the raw rows is exact, so an
            # exact search is one matmul pass with no certificate to fail.
            idx, sq = self.exact(qb, integer=True)
            return idx, np.ldexp(sq.astype(np.float64), -2 * qb.int_shift)

        cand, gmax = self.candidates(qb, self.k_carry, train_cols)
        refined = self.refine(qb, cand)
        mx.eval(cand, refined, gmax, qb.sq)
        idx, sq = _sort_candidates(np.asarray(cand), np.asarray(refined))
        idx, sq = idx[:, : self.k_search], sq[:, : self.k_search]

        if self.k_carry < n_train:
            ok = self.certified(qb, np.asarray(gmax), sq[:, -1])
            if not ok.all():
                bad = np.flatnonzero(~ok)
                idx[bad], sq[bad] = self.resolve(qb.take(mx, bad))

        if state.exponent > 0:
            self.require_resolved(qb, idx, sq)
        return idx, np.ldexp(sq.astype(np.float64), 2 * state.exponent)

    def require_resolved(self, qb: _QueryBlock, idx: np.ndarray, sq: np.ndarray) -> None:
        """Raise if a neighbor returned for ``qb`` is nearer than float32 resolves.

        Only a span scaled down can put a distance below ``_RESOLVED_SQ``, where
        underflow has taken terms from it — all of them on the GPU. An exact
        duplicate of the query is the one neighbor that legitimately sits there.
        Any other would sort first and be returned, so checking the returned
        neighbors is enough; and the certificate, which compares against the k-th
        refined distance, is sound only once that distance is resolved.
        """
        rows, cols = np.nonzero(sq < _RESOLVED_SQ)
        if rows.size == 0:
            return
        train = idx[rows, cols]
        differ = np.any(self.index.X[train] != np.asarray(qb.hi)[rows], axis=1)
        if self.index.X_lo is not None or qb.lo is not None:
            t_lo = 0.0 if self.index.X_lo is None else self.index.X_lo[train]
            q_lo = 0.0 if qb.lo is None else np.asarray(qb.lo)[rows]
            differ |= np.any(t_lo != q_lo, axis=1)
        if differ.any():
            raise UnsupportedInputError(
                "A query row lies nearer to a training row than float32 can resolve at "
                "the training data's scale: the data spans so far from its mean that "
                "the squares of gaps this small underflow. Rescale the data, or compute "
                "these neighbors with scikit-learn.",
                reason="float32-overflow",
            )

    def candidates(self, qb: _QueryBlock, k_carry: int, train_cols: int):
        """Running top-``k_carry`` by Gram value over train tiles, entirely on device.

        Returns the candidate indices and, per row, the largest Gram value kept.
        Every training row left out has a Gram value at least that large — each
        partition and merge keeps the smallest — which is what the certificate
        builds on.
        """
        mx, state = self.mx, self.state
        fns = self.fns
        n_train = self.index.n_samples
        best_idx = None
        best_val = None

        for t_start in range(0, n_train, train_cols):
            t_end = min(t_start + train_cols, n_train)
            gram = qb.sel @ state.sel_t[:, t_start:t_end]
            tile = fns.tile(qb.sq, state.sel_sq[None, t_start:t_end], gram)

            take = min(k_carry, t_end - t_start)
            local, vals = _select_smallest(mx, tile, take, segmented=self.segmented)
            idx = local.astype(state.index_dtype) + t_start

            if best_idx is None:
                best_idx, best_val = idx, vals
            else:
                merged_idx = mx.concatenate([best_idx, idx], axis=1)
                merged_val = mx.concatenate([best_val, vals], axis=1)
                keep = min(k_carry, merged_val.shape[1])
                sel = mx.argpartition(merged_val, kth=keep - 1, axis=1)[:, :keep]
                best_idx = mx.take_along_axis(merged_idx, sel, axis=1)
                best_val = mx.take_along_axis(merged_val, sel, axis=1)

        return best_idx, mx.max(best_val, axis=1)

    def refine(self, qb: _QueryBlock, cand_idx):
        """Recompute the candidates' squared distances directly.

        ``Σ(t − q)²`` from the *untranslated* coordinates, not the expanded
        identity and not the translated copy: translation rounds every coordinate
        to a precision set by its distance from the mean, which for data far from
        its own mean — two clusters, a heavy-tailed column — is far coarser than the
        input itself. Subtracting the raw values rounds once, relative to the
        difference. float64 input also carries its residual, so the difference is
        taken to nearly double precision before it is squared.
        """
        mx, state = self.mx, self.state
        fns = self.fns
        n_rows, k_carry = (int(s) for s in cand_idx.shape)

        # The gather is (rows, k_carry, n_features): with a large k on wide data it
        # dwarfs the distance tile (7.4 GB at k=1000, d=768 before this was
        # budgeted), so it runs in row chunks sized to the budget. Each chunk is
        # evaluated before the next is built: MLX encodes lazily and runs ahead of
        # the GPU, so unevaluated chunks all hold their temporaries at once — lazy
        # chunking alone measured 3.2 GB against a 1 GiB budget.
        gather = _REFINE_WORKSET_FACTOR * k_carry * self.index.n_features * 4
        chunk = max(1, _distance_budget_bytes(self.plan_backend) // max(gather, 1))
        if chunk < n_rows:
            # Finish selection first, so its temporaries are gone before the
            # first chunk's gather is allocated.
            mx.eval(cand_idx)

        parts = []
        for start in range(0, n_rows, chunk):
            end = min(start + chunk, n_rows)
            idx = cand_idx[start:end]
            t = mx.take(state.raw, idx, axis=0)
            q = qb.hi[start:end, None, :]
            if state.raw_lo is None and qb.lo is None:
                sq = fns.pair_sq(t, q, state.scale)
            else:
                t_lo = state.zero if state.raw_lo is None else mx.take(state.raw_lo, idx, axis=0)
                q_lo = state.zero if qb.lo is None else qb.lo[start:end, None, :]
                sq = fns.pair_sq_lo(t, t_lo, q, q_lo, state.scale)
            parts.append(mx.sum(sq, axis=2))
            if chunk < n_rows:
                mx.eval(parts[-1])
        return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=0)

    # ------------------------------------------------------------ certificate

    def certified(self, qb: _QueryBlock, gmax: np.ndarray, r_k: np.ndarray) -> np.ndarray:
        """Rows whose candidate set provably contains their first ``k_search``.

        A training row left out of the candidates has Gram value ``G ≥ gmax``. The
        float32 error of the identity is at most ``C·(‖q‖² + ‖t‖²)``, and the
        reverse triangle inequality gives ``(‖t‖ − ‖q‖)²`` — which is what keeps a
        single far-out training row from making every row uncertifiable. The lower
        bound on its true distance, discounted by refinement's own rounding, must
        exceed the k-th refined distance *strictly*: a row whose excluded
        neighbors might merely tie is not certified, because which tied row makes
        the cut is decided by index, and only an exact search knows the indices.
        """
        state = self.state
        d = self.index.n_features
        q_sq = np.asarray(qb.sq, dtype=np.float64).reshape(-1)
        bound = _excluded_lower_bound(
            q_sq,
            gmax.astype(np.float64),
            state.norms_sorted,
            gram_err=2.0 * (2.0 * _gamma(d + 2) + 8.0 * _U),
            norm_err=2.0 * _gamma(d + 2),
        )
        lower = bound * (1.0 - 2.0 * _gamma(d + 4))
        if qb.raw_norm is not None:
            # The float64 residuals — the rows' and the center's — are themselves
            # rounded, at u² of the raw magnitudes. The center is the training mean,
            # inside the rows' bounding box, so raw_norm_bound covers it as well.
            absolute = math.ldexp(12.0 * _U * _U, -state.exponent) * (
                qb.raw_norm + state.raw_norm_bound
            )
            lower = lower - 2.0 * np.sqrt(np.maximum(bound, 0.0)) * absolute - absolute**2
        return lower > r_k.astype(np.float64)

    # --------------------------------------------------------------- fallback

    def resolve(self, qb: _QueryBlock) -> tuple[np.ndarray, np.ndarray]:
        """Search rows the first pass could not certify: wider, then exactly."""
        mx = self.mx
        n_train = self.index.n_samples
        k_search = self.k_search
        self.counts["widened_rows"] += qb.n
        out_idx = np.empty((qb.n, k_search), dtype=INDEX_DTYPE)
        out_sq = np.empty((qb.n, k_search), dtype=NUMPY_FLOAT)
        pending = np.arange(qb.n)

        k_wide = min(n_train, max(4 * self.k_carry, self.k_carry + _WIDEN_EXTRA))
        if k_wide < n_train:
            plan = resolve_blocks(
                qb.n,
                n_train,
                backend=self.plan_backend,
                k_search=k_wide,
                train_block_size=self.train_block_size,
            )
            still = []
            for start in range(0, qb.n, plan.query_rows):
                rows = np.arange(start, min(start + plan.query_rows, qb.n))
                sub = qb.take(mx, rows)
                cand, gmax = self.candidates(sub, k_wide, plan.train_cols)
                refined = self.refine(sub, cand)
                mx.eval(cand, refined, gmax, sub.sq)
                idx, sq = _sort_candidates(np.asarray(cand), np.asarray(refined))
                idx, sq = idx[:, :k_search], sq[:, :k_search]
                ok = self.certified(sub, np.asarray(gmax), sq[:, -1])
                out_idx[rows[ok]] = idx[ok]
                out_sq[rows[ok]] = sq[ok]
                still.append(rows[~ok])
            pending = np.concatenate(still)

        if pending.size:
            self.counts["exact_rows"] += int(pending.size)
            idx, sq = self.exact(qb.take(mx, pending), integer=False)
            out_idx[pending] = idx
            out_sq[pending] = sq
        return out_idx, out_sq

    def exact(self, qb: _QueryBlock, *, integer: bool) -> tuple[np.ndarray, np.ndarray]:
        """Exact first ``k_search`` against every training row, by composite key.

        Each value is packed with its index into one ``uint64`` —
        ``bits(distance) << 32 | index`` — which orders exactly as ``(distance,
        index)`` because non-negative float32 bit patterns order as their values.
        Partitioning on the key therefore breaks every tie on the index, at every
        tile and every merge, with nothing left to certify.

        With ``integer=True`` the values come from the Gram identity on the raw
        rows, both sides scaled by ``2**qb.int_shift``, which is exact for small
        integers; otherwise from direct differences, exactly as :meth:`refine`
        computes them, so an exact answer and a certified one never disagree about
        a distance. Returns host indices and float32 squared distances — scaled like
        :meth:`refine`'s, or by ``4**int_shift`` when ``integer``.
        """
        mx, state = self.mx, self.state
        fns = self.fns
        n_train = self.index.n_samples
        d = self.index.n_features
        k_search = self.k_search
        budget = _distance_budget_bytes(self.plan_backend)

        if integer:
            plan = resolve_blocks(
                qb.n, n_train, backend=self.plan_backend, k_search=k_search,
                train_block_size=self.train_block_size,
            )
            rows, cols = plan.query_rows, plan.train_cols
            lift = float(2**qb.int_shift)
            q_int = qb.hi * lift if qb.int_shift else qb.hi
            q_sq = mx.sum(q_int * q_int, axis=1, keepdims=True)
        else:
            # A (rows, cols, d) difference per tile; cols wide enough to keep the
            # number of launches down, rows as many as the budget then allows.
            cols = self.train_block_size or 4096
            cols = max(min(k_search, n_train), min(int(cols), n_train))
            rows = max(1, min(qb.n, budget // max(3 * cols * d * 4, 1)))

        keys = []
        for r_start in range(0, qb.n, rows):
            r_end = min(r_start + rows, qb.n)
            best = None
            for t_start in range(0, n_train, cols):
                t_end = min(t_start + cols, n_train)
                if integer:
                    gram = q_int[r_start:r_end] @ state.raw_t[:, t_start:t_end]
                    t_sq = state.raw_sq[None, t_start:t_end]
                    if qb.int_shift:
                        # Both sides lifted by 2**shift: products and sums stay
                        # integers below 2**24, so this is still exact.
                        gram = gram * lift
                        t_sq = t_sq * (lift * lift)
                    vals = fns.tile(q_sq[r_start:r_end], t_sq, gram)
                else:
                    t = state.raw[None, t_start:t_end]
                    q = qb.hi[r_start:r_end, None, :]
                    if state.raw_lo is None and qb.lo is None:
                        sq = fns.pair_sq(t, q, state.scale)
                    else:
                        t_lo = (
                            state.zero if state.raw_lo is None
                            else state.raw_lo[None, t_start:t_end]
                        )
                        q_lo = state.zero if qb.lo is None else qb.lo[r_start:r_end, None, :]
                        sq = fns.pair_sq_lo(t, t_lo, q, q_lo, state.scale)
                    vals = mx.sum(sq, axis=2)

                key = _composite_keys(mx, vals, t_start)
                take = min(k_search, t_end - t_start)
                _, part = _select_smallest(mx, key, take, segmented=self.segmented)
                if best is None:
                    best = part
                else:
                    merged = mx.concatenate([best, part], axis=1)
                    keep = min(k_search, merged.shape[1])
                    sel = mx.argpartition(merged, kth=keep - 1, axis=1)[:, :keep]
                    best = mx.take_along_axis(merged, sel, axis=1)
                # One budget-sized tile in flight at a time; see ``refine``.
                mx.eval(best)
            keys.append(np.array(best))

        packed = np.sort(np.concatenate(keys, axis=0), axis=1)
        idx = (packed & np.uint64(0xFFFFFFFF)).astype(INDEX_DTYPE)
        sq = (packed >> np.uint64(32)).astype(np.uint32).view(NUMPY_FLOAT)
        return idx, sq


def _composite_keys(mx, vals, t_start: int):
    """``bits(vals) << 32 | index`` for a tile whose first column is row ``t_start``.

    32 bits of index, which is why :func:`build_index` refuses ``2**32`` rows. The
    indices are built from 0 and offset, because MLX 0.31's ``arange`` rejects a
    start or stop of ``2**31`` or more.
    """
    index = mx.arange(int(vals.shape[1]), dtype=mx.uint64) + t_start
    return mx.left_shift(mx.view(vals, mx.uint32).astype(mx.uint64), 32) | index[None, :]


def _select_smallest(mx, vals, take: int, *, segmented: bool):
    """Indices and values of the ``take`` smallest entries of each row.

    On the GPU a wide row is partitioned in two stages — each segment first, then
    the survivors — because MLX's GPU ``argpartition`` costs a full sort of the row.
    Exact: every row's global ``take`` smallest are among its segments' ``take``
    smallest. The ragged tail is appended to the survivors whole rather than padded,
    so no padding value can ever tie with, and displace, a real one.
    """
    n_rows, width = vals.shape
    seg = max(_SELECT_SEGMENT, 1 << (4 * take - 1).bit_length())
    if not segmented or seg > _SELECT_SEGMENT_MAX or width < 4 * seg:
        local = mx.argpartition(vals, kth=take - 1, axis=1)[:, :take]
        return local, mx.take_along_axis(vals, local, axis=1)

    n_seg = width // seg
    head = vals[:, : n_seg * seg].reshape(n_rows, n_seg, seg)
    part = mx.argpartition(head, kth=take - 1, axis=2)[:, :, :take]
    cand_val = mx.take_along_axis(head, part, axis=2).reshape(n_rows, n_seg * take)
    offsets = (mx.arange(n_seg, dtype=mx.uint32) * seg)[None, :, None]
    cand_idx = (part + offsets).reshape(n_rows, n_seg * take)

    tail = width - n_seg * seg
    if tail:
        tail_idx = mx.arange(n_seg * seg, width, dtype=mx.uint32)[None, :]
        cand_val = mx.concatenate([cand_val, vals[:, n_seg * seg :]], axis=1)
        cand_idx = mx.concatenate(
            [cand_idx, mx.broadcast_to(tail_idx, (n_rows, tail))], axis=1
        )

    sel = mx.argpartition(cand_val, kth=take - 1, axis=1)[:, :take]
    return mx.take_along_axis(cand_idx, sel, axis=1), mx.take_along_axis(cand_val, sel, axis=1)


def _sort_candidates(idx: np.ndarray, sq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Order each row's candidates by ``(distance, index)``, on the host.

    A lexicographic sort over a ``(block, k_carry)`` array — small enough that host
    ``np.lexsort`` is both simpler and faster than any device trick, and it is the
    one place a total order is imposed on the first pass.
    """
    idx = idx.astype(INDEX_DTYPE, copy=False)
    order = np.lexsort((idx, sq), axis=-1)
    rows = np.arange(idx.shape[0])[:, None]
    return idx[rows, order], sq[rows, order]


def _excluded_lower_bound(
    q_sq: np.ndarray,
    gmax: np.ndarray,
    norms: np.ndarray,
    *,
    gram_err: float,
    norm_err: float,
) -> np.ndarray:
    """Per row, a lower bound on the true distance of every excluded training row.

    For an excluded row ``t`` with norm ``τ = ‖t‖`` two bounds hold:
    ``f(τ) = gmax − C·(‖q‖² + τ²)`` from the Gram error, which falls with ``τ``, and
    ``g(τ) = (τ(1 − η) − ‖q‖(1 + η))² − 16u·(‖q‖² + τ²)`` from the reverse triangle
    inequality, which rises with ``τ`` beyond ``t₁ ≈ ‖q‖``. Splitting the sorted
    training norms at any ``t* ≥ t₁``, rows at or below it are bounded by ``f`` at
    the largest such norm and rows above it by ``g`` at the smallest, so the
    minimum of those two values bounds them all. ``t*`` is placed where ``f`` and
    ``g`` cross, which is the best split; correctness needs only ``t* ≥ t₁``.

    Using the one largest training norm for every row instead would let a single
    outlier make every row uncertifiable.
    """
    C, eta = gram_err, norm_err
    denom = (1.0 - eta) ** 2 - 16.0 * _U
    if not math.isfinite(C) or denom <= 0.0:
        return np.full(q_sq.shape, -np.inf)

    qn = np.sqrt(q_sq)
    a = gmax - C * q_sq
    disc = (1.0 + C) * a - C * q_sq
    cross = (qn + np.sqrt(np.maximum(disc, 0.0))) / (1.0 + C)
    t1 = qn * (1.0 - eta * eta) / denom
    t_star = np.maximum(cross, t1)

    n = norms.shape[0]
    pos = np.searchsorted(norms, t_star, side="right")
    left = norms[np.clip(pos - 1, 0, n - 1)]
    right = norms[np.clip(pos, 0, n - 1)]
    f_left = np.where(pos > 0, a - C * left * left, np.inf)
    g_right = np.where(
        pos < n,
        (right * (1.0 - eta) - qn * (1.0 + eta)) ** 2 - 16.0 * _U * (q_sq + right * right),
        np.inf,
    )
    return np.minimum(f_left, g_right)


def _drop_self(idx: np.ndarray, dist: np.ndarray, *, offset: int):
    """Remove each query's own training index from its neighbor list.

    ``offset`` converts the block-local row number into the global row index. That
    single term is the difference between a correct self-query and the ancestor's,
    which returned a row as its own nearest neighbor whenever the training set
    exceeded one query block.

    A row with no self match — more than ``k`` exact duplicates precede it by index
    — drops its *last* candidate. Under ``(distance, index)`` the row itself sorts
    after every candidate kept, so that leaves the first ``k`` excluding it, and a
    prefix of the answer for any larger ``k``. scikit-learn drops column 0 there,
    but its column 0 is an arbitrary one of the tied duplicates; mlxlearn's is the
    lowest-indexed, the one its own order keeps.
    """
    n_rows, width = idx.shape
    global_rows = (np.arange(n_rows, dtype=idx.dtype) + offset)[:, None]
    mask = idx != global_rows

    missing = np.all(mask, axis=1)
    if np.any(missing):
        mask[missing, -1] = False

    keep = width - 1
    out_idx = idx[mask].reshape(n_rows, keep)
    out_dist = dist[mask].reshape(n_rows, keep)
    return out_idx, out_dist


def _row_sq_norms(mx, a, block: int = 1 << 16):
    """``Σ a²`` per row, in row blocks so no full-size squared temporary exists.

    Each block is evaluated before the next is built; left lazy, MLX encodes them
    all ahead of the GPU and every block's square is alive at once.
    """
    n = int(a.shape[0])
    if n <= block:
        return mx.sum(a * a, axis=1)
    parts = []
    for i in range(0, n, block):
        part = a[i : i + block]
        parts.append(mx.sum(part * part, axis=1))
        mx.eval(parts[-1])
    return mx.concatenate(parts)


def _dyadic_shift(rows: np.ndarray, int_max: float, int_limit: int) -> int | None:
    """The smallest ``s`` making ``rows * 2**s`` integers the exact path can take.

    ``None`` when there is none: a non-dyadic value, or magnitudes that would push
    a product past ``2**24`` once both sides are lifted.
    """
    largest = max(float(np.max(np.abs(rows))) if rows.size else 0.0, int_max)
    for shift in range(_DYADIC_SHIFTS + 1):
        if math.ldexp(largest, shift) > int_limit:
            return None
        lifted = np.ldexp(rows, shift)
        if np.array_equal(lifted, np.rint(lifted)):
            return shift
    return None


def _all_integral(mx, a, block: int = 1 << 16) -> bool:
    """Whether every entry of ``a`` is an integer, a row block at a time.

    Blocked so the rounded copy never exists at full size, and it stops at the first
    block that fails, which for real-valued data is the first.
    """
    for i in range(0, int(a.shape[0]), block):
        part = a[i : i + block]
        if not bool(mx.all(part == mx.round(part)).item()):
            return False
    return True


def _build_state(X: np.ndarray, X_lo: np.ndarray | None, center: np.ndarray, device: str):
    """Upload and derive everything ``device`` needs. Returns ``(state, X, X_lo)``.

    The returned host arrays are read-only zero-copy views of the device arrays;
    the caller swaps them in for the ones it had, which frees those.
    """
    import mlx.core as mx

    fns = _kernels(device)
    n, d = (int(s) for s in X.shape)
    with use_device(device):
        # mx.array copies. This is the index's one private copy of the training
        # rows; nothing below allocates another.
        raw = mx.array(np.ascontiguousarray(X, dtype=NUMPY_FLOAT))
        raw_lo = None if X_lo is None else mx.array(np.ascontiguousarray(X_lo, dtype=NUMPY_FLOAT))
        # The column extremes set the scale, which everything below needs on the
        # host. A small matrix is reduced there in less than a device round trip.
        if n * d <= _HOST_REDUCE_ELEMENTS:
            host = np.asarray(X, dtype=NUMPY_FLOAT)
            col_max = host.max(axis=0)
            col_min = host.min(axis=0)
        else:
            col_max = mx.max(raw, axis=0)
            col_min = mx.min(raw, axis=0)
            mx.eval(raw, col_max, col_min, *([] if raw_lo is None else [raw_lo]))

        c64 = np.asarray(center, dtype=np.float64)
        hi_col = np.asarray(col_max, dtype=np.float64)
        lo_col = np.asarray(col_min, dtype=np.float64)
        spread = float(max(np.max(np.abs(hi_col - c64)), np.max(np.abs(lo_col - c64))))
        if not math.isfinite(spread):
            raise ValueError("The training data contains NaN or infinity.")
        if spread > _SPAN_LIMIT:
            # The remedy is advice: a patched estimator falls back on this at fit,
            # and reports the limit without telling its caller to use scikit-learn.
            raise UnsupportedInputError(
                f"The training data spans {spread:.3g} from its mean, too wide for "
                "float32 to subtract two of its coordinates.",
                reason="float32-overflow",
                advice="Rescale it, or compute these neighbors with scikit-learn.",
            )
        # Scale by an exact power of two, and only where float32 needs it: data
        # around 1e19, whose squares overflow float32, and data around 1e-20, whose
        # squares underflow it (and which the GPU flushes to zero), are both brought
        # back to where float32 squares them faithfully. Scaling by 2**k is exact,
        # and ordinary data is not scaled at all.
        exponent = _span_exponent(spread, d)
        scale = mx.array(math.ldexp(1.0, -exponent), dtype=mx.float32)
        c_hi = c64.astype(NUMPY_FLOAT)
        c_lo = (c64 - c_hi).astype(NUMPY_FLOAT)
        center_dev = mx.array(c_hi)
        center_lo = mx.array(c_lo) if c_lo.any() else None
        zero = mx.array(0.0, dtype=mx.float32)

        sel = _translate(fns, raw, raw_lo, center_dev, center_lo, scale, zero)
        sel_sq = _row_sq_norms(mx, sel)
        # Everything that goes into the state is evaluated here, the transposes
        # included: an unevaluated node belongs to this thread's stream, and a query
        # from another thread that forces it fails with "There is no Stream(gpu, N)
        # in current thread".
        sel_t = sel.T
        pending = [raw, sel, sel_t, sel_sq, *([] if raw_lo is None else [raw_lo])]
        # The certificate's sorted norms. MLX's CPU sort took a third of a CPU
        # build, four times NumPy's; the float32 square roots are the same either way.
        norms = mx.sort(mx.sqrt(sel_sq)) if device == "gpu" else None
        if norms is not None:
            pending.append(norms)

        # Small integers — binary, one-hot, counts — make the raw Gram identity
        # exact in float32: every product and partial sum is an integer below 2**24.
        # Such data is also where ties are heaviest, and an exact search needs no
        # certificate, so it gets its own path.
        max_abs = float(max(np.max(np.abs(hi_col)), np.max(np.abs(lo_col)))) if n else 0.0
        int_limit = int(math.isqrt((1 << 24) // (4 * d))) if d else 0
        raw_t = raw_sq = None
        if raw_lo is None and 0 < int_limit and max_abs <= int_limit:
            # Real-valued data within the bound — standardized features, say —
            # nearly always shows it in its first rows, which the host can check
            # without a pass over the device copy or a synchronization.
            head = np.asarray(X[:256], dtype=NUMPY_FLOAT)
            if np.array_equal(head, np.rint(head)) and _all_integral(mx, raw):
                raw_t = raw.T
                raw_sq = _row_sq_norms(mx, raw)
                pending += [raw_t, raw_sq]
            else:
                int_limit = 0
        else:
            int_limit = 0
        mx.eval(*pending)

    state = _DeviceState(
        raw=raw,
        raw_lo=raw_lo,
        sel=sel,
        sel_t=sel_t,
        sel_sq=sel_sq,
        center=center_dev,
        center_lo=center_lo,
        scale=scale,
        zero=zero,
        exponent=exponent,
        spread=spread,
        norms_sorted=(
            np.sort(np.sqrt(np.asarray(sel_sq))) if norms is None else np.asarray(norms)
        ).astype(np.float64),
        raw_norm_bound=float(np.sqrt(np.sum(np.maximum(np.abs(hi_col), np.abs(lo_col)) ** 2))),
        int_limit=int_limit,
        int_max=max_abs,
        raw_t=raw_t,
        raw_sq=raw_sq,
        index_dtype=mx.int32 if n < 2**31 else mx.int64,
    )
    x_view = _readonly_view(raw)
    lo_view = None if raw_lo is None else _readonly_view(raw_lo)
    return state, x_view, lo_view


def _readonly_view(array) -> np.ndarray:
    """A zero-copy NumPy view of an evaluated MLX array, write-protected.

    MLX arrays are immutable, and the device state assumes it; a writeable view
    would let a caller change the index underneath its own derived arrays.
    """
    view = np.asarray(array)
    view.flags.writeable = False
    return view


def _split_float64(X: np.ndarray, block: int = 1 << 16) -> tuple[np.ndarray, np.ndarray | None]:
    """``X`` as a float32 value plus a float32 residual, ``hi + lo ≈ X`` to ~u².

    Row-blocked, so the float64 difference never exists at full size. The residual
    is ``None`` when ``X`` is exactly representable in float32 — integers, binary
    features, data that was float32 all along.
    """
    hi = np.asarray(X, dtype=NUMPY_FLOAT, order="C")
    lo = np.empty_like(hi)
    for start in range(0, X.shape[0], block):
        stop = start + block
        lo[start:stop] = X[start:stop] - hi[start:stop]
    return hi, (lo if lo.any() else None)


def build_index(X: np.ndarray, *, backend: str, device: str | None = None) -> NeighborIndex:
    """Build a :class:`NeighborIndex` from a training matrix, on ``device`` if given.

    float32 input is uploaded as given. The upload is the copy — the index then
    keeps a read-only view of its own device buffer, never the caller's array —
    because an index that aliases the caller's array silently changes its own
    answers when that array is mutated. Wider input (float64, integers) is split
    into a float32 value and a float32 residual, and translated by its float64
    mean, carried as a float32 pair the same way, so the float32 cast does not
    quantize away what the caller supplied: Unix timestamps collapse to 128-second
    steps in float32, but not here.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-dimensional; got shape {X.shape}.")
    if X.shape[0] > _MAX_TRAIN_ROWS:
        raise UnsupportedInputError(
            f"The training data has {X.shape[0]} rows; the neighbor index addresses at "
            f"most {_MAX_TRAIN_ROWS}.",
            advice="Compute these neighbors with scikit-learn.",
        )
    if X.dtype == NUMPY_FLOAT:
        hi = np.ascontiguousarray(X)
        lo = None
        center = hi.mean(axis=0, dtype=np.float64).astype(NUMPY_FLOAT)
    else:
        X64 = np.asarray(X, dtype=np.float64)
        hi, lo = _split_float64(X64)
        center = X64.mean(axis=0)
    index = NeighborIndex(X=hi, center=center, backend=backend, X_lo=lo, device=device)
    _require_resolvable(index, device)
    return index


def _require_resolvable(index: NeighborIndex, device: str | None) -> None:
    """Refuse, at fit, training rows whose own neighbors float32 cannot resolve.

    Only reachable when the span had to be scaled down: a few values far larger than
    the rest — an unmasked fill value, 1e20 or the NetCDF default 9.97e36, among
    unit-scale rows — leave the rest's gaps too small to square at that scale. A
    query would then raise at inference, where nothing can fall back, so a sample
    of the training rows is searched here instead, and a fit whose rows sit within
    ``_RESOLVED_MARGIN`` of the limit is a capability mismatch: a directly imported
    estimator raises, and a patched one falls back to scikit-learn.
    """
    state = index._ensure_device(device)
    n = index.n_samples
    if state.exponent <= 0 or n < 2:
        return
    rows = np.unique(np.linspace(0, n - 1, min(n, _PROBE_ROWS)).astype(np.intp))
    probe = index.X[rows]
    if index.X_lo is not None:
        probe = probe.astype(np.float64) + index.X_lo[rows]
    try:
        # Each row finds itself, or a duplicate, first; the second is its nearest.
        dist, idx = index.query(probe, 2, device=device)
    except UnsupportedInputError:
        unresolved = True  # a neighbor already below the limit itself
    else:
        near = idx[:, 1]
        close = np.ldexp(np.square(dist[:, 1]), -2 * state.exponent) < (
            _RESOLVED_SQ * _RESOLVED_MARGIN
        )
        differ = np.any(index.X[near] != index.X[rows], axis=1)
        if index.X_lo is not None:
            differ |= np.any(index.X_lo[near] != index.X_lo[rows], axis=1)
        unresolved = bool(np.any(close & differ))
    if unresolved:
        raise UnsupportedInputError(
            f"The training data spans {state.spread:.3g} from its mean, too wide for float32 to "
            "also resolve the distances between its nearest rows: their squares underflow "
            "at that scale.",
            reason="float32-overflow",
            advice="A fill or sentinel value left in the data does this; mask it, or "
            "rescale the data, or compute these neighbors with scikit-learn.",
        )
