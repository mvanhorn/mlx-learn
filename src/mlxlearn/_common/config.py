"""Configuration: a small public surface over a larger private one.

Two layers, on purpose.

:class:`Config` is public and deliberately tiny — six options that change what a
user observes: which device to use, what to do when mlxlearn cannot serve a
request, what array type comes back, whether results are reproducible, the seed,
and whether diagnostics are recorded.

:class:`Tuning` is private. Crossover thresholds, block sizes, cache fractions and
solver limits are implementation details that will move as the benchmarks improve;
promoting them to public options would freeze today's implementation into the API.
They are overridable through ``MLXLEARN_*`` environment variables for CI and
debugging, and that is the whole supported surface for them.

**Where a setting reaches.** Three scopes, and each one exists because a user
reasonably expects it:

* :func:`set_config` changes the *process-wide* default. Every thread sees it —
  including the worker threads scikit-learn's ``n_jobs`` machinery runs fits on
  under joblib's threading backend. A per-thread ``set_config`` made
  ``fallback_policy="raise"`` pass vacuously the moment a ``GridSearchCV`` went
  parallel, which is the one place an assertion of full acceleration is worth
  making.
* :func:`config_context` overrides the fields it names for the calling *context*
  only (a :class:`contextvars.ContextVar`), so a library that sets a policy inside
  it cannot leak it into a thread it did not start. ``asyncio`` tasks and
  ``asyncio.to_thread`` inherit it, as they inherit any context variable; plain
  threads and thread pools, joblib's included, do not.
* A separate *process* — joblib's default ``loky`` backend, ``multiprocessing`` —
  imports mlxlearn afresh and starts from the environment: ``MLXLEARN_DEVICE``,
  ``MLXLEARN_FALLBACK_POLICY``, ``MLXLEARN_DIAGNOSTICS`` and so on, one per
  :class:`Config` field. Nothing a parent sets at runtime can reach a worker that
  is already running, so to govern process workers set those variables before
  the workers start. (To *assert* that nothing falls back under ``n_jobs``, a
  ``warnings.filterwarnings("error", category=MLXLearnFallbackWarning)`` also
  works: scikit-learn forwards warning filters to its workers.)
"""

from __future__ import annotations

import numbers
import os
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, fields, replace
from typing import Any, Literal, get_args

import numpy as np

__all__ = [
    "Config",
    "Tuning",
    "get_config",
    "set_config",
    "config_context",
    "get_tuning",
    "tuning_context",
    "reset_config",
]

Device = Literal["auto", "gpu", "cpu"]
FallbackPolicy = Literal["warn", "raise", "silent"]
OutputType = Literal["numpy", "input", "mlx"]
ForceBackend = Literal["", "mlx", "cpu", "sklearn"]

_SEED_LIMIT = 2**32


# --------------------------------------------------------------------------------------
# public configuration
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Config:
    """User-visible configuration. Immutable; replace it, do not mutate it.

    Attributes
    ----------
    device : {"auto", "gpu", "cpu"}
        Which MLX device runs the work that dispatch has sent to MLX. ``"auto"``
        prefers the GPU and falls back to the CPU when no Metal device is usable.
        It does not override size dispatch: a problem below the measured
        crossover still goes to mlxlearn's CPU path (Layer 1) or to scikit-learn
        (patched), whatever the device. ``device="cpu"`` is the same float32
        algorithm on the CPU, not scikit-learn's float64 one. The names are
        MLX's, not Metal's: there is deliberately no ``"mps"``.
    fallback_policy : {"warn", "raise", "silent"}
        What patched mode does on a *capability* mismatch. ``"warn"`` falls back
        and warns at most once per estimator class and reason; ``"raise"`` turns
        any fallback into an error, which is how you assert that a workload is
        fully accelerated; ``"silent"`` falls back quietly, for production and
        agent harnesses where a warning stream is just noise. Unexpected backend
        failures raise under every policy.
    output_type : {"numpy", "input", "mlx"}
        What array type estimators return. 0.1.0 supports ``"numpy"`` only;
        ``"input"`` and ``"mlx"`` are reserved for the 0.2.x experimental MLX
        boundary and are rejected today rather than silently ignored.
    deterministic : bool
        When true, mlxlearn takes reproducible paths — fixed reduction orders,
        seeded initialization, stable tie-breaking — even where a faster
        nondeterministic path exists. Every 0.1.x solver is deterministic and has
        no faster nondeterministic variant, so today this option has no
        observable effect: results are reproducible either way.
    random_state : int or None
        Seed for mlxlearn's own RNG, an int in ``[0, 2**32)``. Estimator-level
        ``random_state`` parameters take precedence over it. No 0.1.x estimator
        draws random numbers, so today it has no observable effect; it does not
        seed scikit-learn either, so a patched estimator that falls back uses its
        own ``random_state`` parameter.
    diagnostics : bool
        Whether backend events are recorded. Recording is cheap and it is what
        makes a fallback debuggable after the fact, so it defaults on.
    """

    device: Device = "auto"
    fallback_policy: FallbackPolicy = "warn"
    output_type: OutputType = "numpy"
    deterministic: bool = True
    random_state: int | None = None
    diagnostics: bool = True

    def __post_init__(self) -> None:
        _validate_choice("device", self.device, get_args(Device))
        _validate_choice("fallback_policy", self.fallback_policy, get_args(FallbackPolicy))
        _validate_choice("output_type", self.output_type, get_args(OutputType))
        if self.output_type != "numpy":
            raise ValueError(
                f"output_type={self.output_type!r} is reserved for the 0.2.x experimental "
                "MLX array boundary and is not implemented in this release. "
                "mlxlearn 0.1.0 takes NumPy in and returns NumPy out."
            )
        # Normalized, not merely checked: a NumPy bool or integer is the value the
        # user meant, and storing the Python type keeps equality and pickles plain.
        object.__setattr__(self, "deterministic", _as_bool("deterministic", self.deterministic))
        object.__setattr__(self, "diagnostics", _as_bool("diagnostics", self.diagnostics))
        object.__setattr__(self, "random_state", _as_seed(self.random_state))


def _validate_choice(name: str, value: object, allowed: tuple[Any, ...]) -> None:
    if value not in allowed:
        options = ", ".join(repr(a) for a in allowed)
        raise ValueError(f"{name} must be one of {options}; got {value!r}")


def _describe(value: object) -> str:
    kind = type(value)
    module = "" if kind.__module__ == "builtins" else f"{kind.__module__}."
    return f"{value!r} ({module}{kind.__qualname__})"


def _as_bool(name: str, value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise TypeError(f"{name} must be a bool, got {_describe(value)}")


def _as_seed(value: object) -> int | None:
    if value is None:
        return None
    # bool is an Integral; a seed of True is a bug, not a seed of 1.
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise TypeError(f"random_state must be an int or None, got {_describe(value)}")
    seed = int(value)
    if not 0 <= seed < _SEED_LIMIT:
        raise ValueError(f"random_state must be in [0, 2**32), got {seed}")
    return seed


# --------------------------------------------------------------------------------------
# private tuning
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Tuning:
    """Private tuning knobs. Not public API; may change in any release.

    Every default here is a *measurement*, or a placeholder waiting for one. The
    crossover thresholds are the sizes below which stock scikit-learn wins, taken
    from ``docs/benchmarks.md``; dispatch reads them so that "we are never slower"
    is enforced by handing small problems away rather than by hoping.
    """

    # ---- dispatch crossovers: below these, stock scikit-learn is used ----------------
    #
    # Measured, not guessed: the MLX path patched dispatch would run against stock
    # scikit-learn in interleaved, order-alternated pairs (benchmarks/_harness.compare:
    # a garbage collection before every timed call, mx.synchronize inside it, so every
    # MLX call pays to wake an idled GPU), median of 5-9, on an Apple M4 Max (16 cores,
    # 128 GiB, macOS 26.6.2) with MLX 0.32.2, against scikit-learn 1.9.0 (Python
    # 3.13.3, NumPy 2.5.2) and 1.7.2 (Python 3.10.19, NumPy 2.2.6). A threshold routes
    # to MLX only where every configuration measured won, or tied within noise, on
    # both versions. Each value below names the calibration table in
    # docs/benchmarks.md that justifies it, and `python -m benchmarks.calibrate`
    # regenerates that table (the grid of the same name).
    #
    # KNN, k=10, Gaussian data. Both floors bind in both layers (see
    # MLXNeighborsMixin._below_crossover), because a fit cannot see how it will be
    # queried and each floor guards a different batch size (table `knn-rows`: 256 to
    # 1024 features, 2048-131072 rows, every batch of 1-4096 queries). Below 4096
    # rows a 4096-query batch lost at every width (0.44x-0.79x at 2048 rows). Below
    # 6e6 elements batches of 1-32 queries lost at 4096-8192 rows (0.70x-0.95x), and
    # at 4 194 304 elements, the floor this one replaced, single rows only tied
    # (0.90x-0.99x). Above both, at 256-1024 features, every batch won on both
    # versions or tied (1.02x-8.5x).
    knn_min_train_samples: int = 4_096
    knn_min_work: int = 6_000_000  # n_train * n_features
    #
    # Patched mode only (MLXNeighborsMixin._patched_handoff), two more rules. A fit
    # with fewer than knn_min_features features goes to scikit-learn at any size: to
    # its KD-tree where algorithm='auto' would build one (at most 15 features; reason
    # "tree-wins"; tables `knn-tree`, `knn-tree-lowrank`), else to its brute force
    # (tables `knn-narrow-brute`, `knn-narrow`). Re-measured at a million rows on
    # Gaussian, clustered and uniform data (table `knn-narrow`), 16-48 features lost
    # a batch on one version or both — 16 queries at 0.73x-0.84x, 1024 at
    # 0.62x-0.79x — which the `knn` table, measured on Gaussian data alone, had
    # shown as wins; from 64 features every batch won or tied (0.94x at worst).
    # 0 disables the rule.
    knn_min_features: int = 64
    #
    # And a fit with fewer than knn_wide_features features needs
    # knn_narrow_min_samples rows (table `knn`: 16-256 features, 12 000-1.6 million
    # rows, every batch of 1-4096 queries). An idled GPU runs a call of a few
    # milliseconds' work up to two or three times slower, so at 16-128 features
    # batches of 8-512 queries lost on one version or both at 42 of the 50 sizes
    # measured below a million rows (down to 0.44x; 400 000 x 64 and 400 000 x 96
    # among them), while from 1 000 000 rows at 64-128 features every batch won or
    # tied (tables `knn`, `knn-narrow`). At 256 features every batch won or tied
    # from 12 000 rows. The previous floors alone sent 180 000 x 32 to MLX, where
    # 32-query batches ran 0.57x-0.64x.
    knn_wide_features: int = 256
    knn_narrow_min_samples: int = 1_000_000
    #
    # All of the above is calibrated on Gaussian data. Patched mode also reads the
    # data (MLXNeighborsMixin._patched_data_handoff): the fit searches 64 of its own
    # rows with the first float32 pass alone and hands the fit to scikit-learn when
    # more than this fraction could not be certified — rows every query like them
    # would search again, wider and then exactly (table `knn-data`). Gaussian,
    # clustered (std 1) and uniform data left none uncertified at every shape
    # measured, and won or tied; one column in raw units beside standardized ones,
    # rows repeated about 20 times, and 100 clusters of std 0.1 left 59-64 of 64,
    # and lost at every shape the size rules send to MLX, down to 0.16x.
    # Integer-valued data takes an exact path with nothing to certify; narrower than
    # knn_wide_features it goes to scikit-learn too: at a million rows and 16 or 32
    # features it lost batches of 256 and 1024 queries (0.48x-0.88x), while from 256
    # features it kept up (64-255 features were not measured on MLX).
    knn_probe_max_uncertified: float = 0.125
    #
    # Logistic regression, max_iter=200 (see LogisticRegression._below_crossover).
    # Every fit needs both floors. Multinomial fits won at every width measured from
    # 60000 rows, on overlapping blobs (table `logreg-multi`: 3, 5 and 10 classes, 1
    # to 512 features, 1.50x-8.6x on both versions) and on well-separated ones that
    # scikit-learn fits in one iteration (table `logreg-separable`: 1.00x-4.7x); at
    # 20000 rows they were mixed (0.78x-2.3x), and at 5000 lost or tied at every
    # width but 512 (0.37x-0.91x). The work floor is the smallest fit measured to win, binary
    # 100000 x 1; above the row floors it does not bind.
    logreg_min_samples: int = 60_000
    logreg_min_work: int = 100_000  # n_samples * n_features * weight vectors
    #
    # A binary fit also needs logreg_binary_min_samples rows and at most
    # logreg_binary_max_features features. On the benchmark's binary target MLX won
    # from 100000 rows at every width up to 1024 (table `logreg-binary`, 1.30x-6.2x),
    # but a fit that converges in an iteration or two is mostly MLX's fixed cost —
    # the float32 copy of X and the device round trips — and labels carrying no
    # signal do exactly that (table `logreg-labels`): at 100000 rows random and noisy
    # labels lost on 1.7.2 at 4-256 features (0.61x-0.88x), and random ones on 1.9.0
    # at 4 and 256 features (0.81x, 0.90x); from 200000 rows they won
    # or tied up to 64 features (0.97x at worst), and at 256 features random labels
    # stood at 0.96x-1.17x, too close to call.
    logreg_binary_min_samples: int = 200_000
    logreg_binary_max_features: int = 64
    #
    # SVC: at the default kernel parameters the axis that matters is *width* (see
    # SVC._below_crossover). Per SMO iteration mlxlearn costs about the same at any
    # width while LIBSVM pays O(n*d) per kernel row it misses. Table `svc`: binary
    # rbf and poly fits, default gamma and degree, at C = 1 and 10 everywhere, 0.1
    # and 100 around the floors. At 16-64 features both lost at C = 10 and 100 on
    # both versions (0.15x-0.85x), and poly at C = 1 up to 32 features. From 128
    # features they won at every C — once the fit had rows: at 128 features
    # 1 000-2 000 rows lost at C = 10 and above on one version or both
    # (0.61x-0.95x), and from 3 000 rows every kernel and C won or tied. The work
    # floor is that corner, 3000 x 128, which also keeps 1000 x 256 (a 1.04x win at
    # worst) and lets 1500 x 256 (1.36x-5.0x) and 1024 x 512 through. Above all
    # three floors every binary fit measured at the default kernel parameters won or
    # tied on both versions, 1.04x-8.4x. The linear kernel (table `svc-linear`) and
    # the sigmoid kernel (table `svc-sigmoid`: won at C = 0.1, lost at C = 10 and
    # 100 nearly everywhere, down to 0.06x) are excluded at every size; for rbf and
    # poly, patched mode also reads the kernel parameters (below).
    svc_min_features: int = 128
    svc_min_samples: int = 1_024
    svc_min_work: int = 384_000  # n_samples * n_features
    #
    # One-vs-one fits K(K-1)/2 subproblems of 2n/K rows on average, each paying a
    # fixed Gram dispatch and its own Python SMO loop, so many small classes lose
    # even when the total clears the floors above. The mean pair must clear the
    # binary work floor itself: pairs of 2000 x 128 — the corner svc_min_work
    # excludes for two classes — lost on overlapping classes at C = 10 and 100 on
    # 1.7.2 (table `svc-multi-overlap`, 0.91x-0.92x), while every pair from 3000 x 128
    # up won or tied (1.01x-3.5x); on separated blobs (table `svc-multi`) 256 x 512
    # (K=16) ran 0.73x-0.81x and 500 x 256 (K=32) 0.70x-0.77x. What the floor gives
    # up are pairs of 512 x 512 and 1000 x 256, which won (on separated blobs
    # 1.28x-1.39x and 1.17x-1.21x; 1000 x 256 on overlapping classes 1.09x-2.0x).
    # This floor is on the *mean* pair's work, (2n/K) * n_features, which ignores
    # class imbalance by construction.
    svc_min_pair_work: int = 384_000
    #
    # Patched mode also reads the kernel parameters (SVC._kernel_handoff; table
    # `svc-params`, binary fits at shapes the floors above send to MLX, C = 1-100).
    # A kernel close to the linear one takes many SMO iterations per row over rows
    # LIBSVM has cached, and loses at any width. rbf needs gamma of at least
    # svc_min_gamma_ratio times scikit-learn's "scale" value, gamma * d * Var(X):
    # from 0.03 to 0.5 times it lost at C = 10 or 100 at five of the six shapes
    # (down to 0.30x), at 0.8 it won or tied everywhere (1.01x at worst, 3000 x 128
    # on 1.7.2), and from 1 — the default — up to 10 it won. Below that it needs C
    # at most svc_low_gamma_max_c: at C = 1 every gamma below it won at every
    # shape (2.1x-11x). The poly kernel needs degree svc_poly_min_degree or more and
    # coef0 = 0: degree 2 lost at C = 10 and 100 at 128 features (0.43x-0.63x), and
    # coef0 = 1 at degree 2 or 3 lost at 128 features (down to 0.47x) and, at degree
    # 2, at 1500 x 256 (0.74x-0.94x); degree 3 and 4 at coef0 = 0 won or tied
    # everywhere.
    svc_min_gamma_ratio: float = 0.8
    svc_low_gamma_max_c: float = 1.0
    svc_poly_min_degree: int = 3
    #
    # Not a dispatch rule but where a fitted SVC computes (SVC._pairwise_decision): a
    # predict or decision_function of at most this much work — query rows x support
    # vectors x features — is evaluated on the host, in NumPy float32, from the same
    # operand; larger batches on the fitted device. What the device charges a batch
    # of a few rows is the call, 0.64-0.98 ms once the GPU has idled: with this at
    # 0, a single row lost to LIBSVM at five of the six fits of `svc-predict` on one
    # version or both (0.27x-0.78x), and 2-4 rows at four of them; only at 6000 x 512,
    # where LIBSVM's own predict is slowest, did the device tie or win. On the host
    # (table `svc-predict`) single rows tied or won
    # (1.06x-2.7x) and every larger batch won, on both versions. The host was faster
    # than an idle GPU up to about 5e7 of this work at 128 features and further out
    # at wider data; 2**25 is below the crossover at every shape measured.
    svc_host_inference_work: int = 33_554_432

    # ---- blocking ---------------------------------------------------------------------
    distance_block_bytes: int = 256 * 1024 * 1024
    distance_block_bytes_max: int = 1024 * 1024 * 1024
    query_block_rows: int = 1024
    query_block_align: int = 128
    cpu_distance_block_bytes: int = 128 * 1024 * 1024
    cpu_query_block_rows: int = 256

    # ---- solvers ----------------------------------------------------------------------
    lbfgs_history: int = 10
    lbfgs_max_line_search: int = 32
    smo_kernel_cache_fraction: float = 0.05  # of physical memory
    smo_kernel_cache_min_bytes: int = 32 * 1024 * 1024
    smo_kernel_cache_max_bytes: int = 2 * 1024 * 1024 * 1024
    smo_max_iter_factor: int = 100  # max_iter=-1 -> factor * n_samples, bounded below
    smo_min_iter_budget: int = 10_000_000

    # ---- memory -----------------------------------------------------------------------
    #
    # MLX returns freed buffers to its own cache rather than to the OS, and its
    # default cache limit is the memory limit (~95% of RAM). One 200k x 128
    # kneighbors call left 10 GiB cached with nothing alive; fits whose n_train
    # varied kept adding, to 50 GiB after six. So a top-level MLX operation that
    # grew the cache by more than this fraction of physical memory releases it on
    # the way out (mx.clear_cache frees only idle buffers and changes no setting).
    # The cost is reallocation on the next call of the same shape: a repeated
    # 150k x 64 / 6k-query kneighbors measured 470 -> 618 ms with the release
    # forced every call (it grows the cache by 7.5 GiB). 1/16 of RAM leaves that
    # case alone on a 128 GB machine (8 GiB threshold) and releases it on a 16 GB
    # one (1 GiB), where keeping it would pin half the machine. 0 disables it.
    mlx_cache_release_fraction: float = 1 / 16

    # ---- behavior ---------------------------------------------------------------------
    #: Force a specific backend regardless of size. For tests and bisection only.
    force_backend: ForceBackend = ""
    #: Ring-buffer size for diagnostics. The buffer is shared by every thread, so
    #: this is read from the process-wide default only (MLXLEARN_DIAGNOSTICS_CAPACITY,
    #: at import and on reset_config); tuning_context cannot resize it.
    diagnostics_capacity: int = 512

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, int) and not isinstance(value, bool) and value < 0:
                raise ValueError(f"tuning value {f.name} must be non-negative, got {value}")
        # Zero here is not "off", it is a division by zero or an empty block
        # somewhere deep in a kernel. The error belongs at configuration time,
        # where it can name the knob.
        for name in _AT_LEAST_ONE:
            if getattr(self, name) < 1:
                raise ValueError(
                    f"tuning value {name} must be at least 1, got {getattr(self, name)}"
                )
        if not 0.0 < self.smo_kernel_cache_fraction <= 1.0:
            raise ValueError(
                "smo_kernel_cache_fraction must be in (0, 1], got "
                f"{self.smo_kernel_cache_fraction}"
            )
        # Written so that NaN fails the comparison and is rejected too.
        for name in ("mlx_cache_release_fraction", "knn_probe_max_uncertified"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {getattr(self, name)}")
        if self.force_backend not in get_args(ForceBackend):
            # Not lower-cased: "MLX" is as much a typo as "gpu", and a forced
            # backend that silently meant something else defeats the bisection
            # the knob exists for. The GPU is chosen with Config.device.
            raise ValueError(
                "force_backend must be one of '', 'mlx', 'cpu', 'sklearn' (the MLX "
                "GPU backend is 'mlx'; pick the device with mlxlearn's device option); "
                f"got {self.force_backend!r}"
            )


_AT_LEAST_ONE = (
    "distance_block_bytes",
    "distance_block_bytes_max",
    "query_block_rows",
    "query_block_align",
    "cpu_distance_block_bytes",
    "cpu_query_block_rows",
    "lbfgs_history",
    "lbfgs_max_line_search",
)

_ENV_PREFIX = "MLXLEARN_"


def _tuning_from_env(base: Tuning) -> Tuning:
    """Apply ``MLXLEARN_*`` overrides on top of ``base``.

    A malformed value is an error, not a silent fallback to the default: an
    environment variable that quietly does nothing is worse than no override at
    all, because you cannot tell the two apart from the outside.
    """
    overrides: dict[str, Any] = {}
    for f in fields(Tuning):
        key = _ENV_PREFIX + f.name.upper()
        raw = os.environ.get(key)
        if raw is None:
            continue
        try:
            if f.type is int or f.type == "int":
                overrides[f.name] = int(raw)
            elif f.type is float or f.type == "float":
                overrides[f.name] = float(raw)
            else:
                overrides[f.name] = raw
        except ValueError as exc:
            raise ValueError(f"{key}={raw!r} is not a valid value for {f.name}") from exc
    if not overrides:
        return base
    try:
        return replace(base, **overrides)
    except ValueError as exc:
        names = ", ".join(_ENV_PREFIX + name.upper() for name in sorted(overrides))
        raise ValueError(f"invalid tuning override in the environment ({names}): {exc}") from exc


_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


def _config_from_env() -> Config:
    """The process-wide default configuration: :class:`Config` plus ``MLXLEARN_*``.

    This is what a fresh process — a loky or multiprocessing worker included —
    starts from. Same rule as tuning: a malformed value is an error.
    """
    overrides: dict[str, Any] = {}
    for f in fields(Config):
        key = _ENV_PREFIX + f.name.upper()
        raw = os.environ.get(key)
        if raw is None:
            continue
        text = raw.strip()
        if f.name in ("deterministic", "diagnostics"):
            if text.lower() in _TRUE:
                overrides[f.name] = True
            elif text.lower() in _FALSE:
                overrides[f.name] = False
            else:
                raise ValueError(f"{key}={raw!r} is not a boolean (use 1/0, true/false)")
        elif f.name == "random_state":
            if text.lower() in ("", "none"):
                overrides[f.name] = None
            else:
                try:
                    overrides[f.name] = int(text)
                except ValueError as exc:
                    raise ValueError(f"{key}={raw!r} is not an integer seed") from exc
        else:
            overrides[f.name] = text
    try:
        return Config(**overrides)
    except (TypeError, ValueError) as exc:
        names = ", ".join(_ENV_PREFIX + name.upper() for name in sorted(overrides))
        raise ValueError(f"invalid configuration in the environment ({names}): {exc}") from exc


# --------------------------------------------------------------------------------------
# state: process-wide defaults, context-local overrides
# --------------------------------------------------------------------------------------
#
# Two immutable values and a lock. set_config swaps the default atomically; the
# overrides live in context variables, so they are per thread and per asyncio
# task without any per-thread copy of the defaults that could go stale (a pool
# thread that had copied the default on first use would never see a later
# set_config).


class _ConfigScope:
    """One active :func:`config_context`: the fields it overrides, merged with outer ones.

    The resolved :class:`Config` is cached against the default it was built on,
    so ``get_config`` stays a couple of attribute reads, and a ``set_config``
    made while the scope is active still shows through for every field the
    scope does not override.
    """

    __slots__ = ("overrides", "_cache")

    def __init__(self, overrides: dict[str, Any], default: Config) -> None:
        self.overrides = overrides
        self._cache = (default, replace(default, **overrides))

    def resolve(self, default: Config) -> Config:
        base, resolved = self._cache
        if base is not default:
            resolved = replace(default, **self.overrides)
            self._cache = (default, resolved)
        return resolved


_lock = threading.Lock()
_default_config: Config = _config_from_env()
_default_tuning: Tuning = _tuning_from_env(Tuning())
_config_scope: ContextVar[_ConfigScope | None] = ContextVar("mlxlearn_config", default=None)
_tuning_scope: ContextVar[Tuning | None] = ContextVar("mlxlearn_tuning", default=None)


def _check_config_names(kwargs: dict[str, Any]) -> None:
    known = {f.name for f in fields(Config)}
    unknown = set(kwargs) - known
    if unknown:
        raise TypeError(
            f"unknown configuration option(s): {', '.join(sorted(unknown))}. "
            f"Valid options are: {', '.join(sorted(known))}."
        )


def get_config() -> Config:
    """Return the configuration in effect here.

    That is the process-wide default set by :func:`set_config`, with the fields
    named by any enclosing :func:`config_context` overridden.
    """
    scope = _config_scope.get()
    default = _default_config
    return default if scope is None else scope.resolve(default)


def set_config(**kwargs: Any) -> Config:
    """Update the process-wide default configuration and return the configuration in effect.

    Only the fields named are changed. Unknown field names are an error rather
    than being stored and ignored. The new default reaches every thread,
    including scikit-learn's ``n_jobs`` worker threads; process workers start
    from the ``MLXLEARN_*`` environment instead (see the module docstring).
    Inside a :func:`config_context`, the fields that context overrides keep its
    value until it exits.

    Examples
    --------
    >>> import mlxlearn
    >>> _ = mlxlearn.set_config(fallback_policy="raise")
    >>> mlxlearn.get_config().fallback_policy
    'raise'
    >>> _ = mlxlearn.set_config(fallback_policy="warn")
    """
    global _default_config
    _check_config_names(kwargs)
    with _lock:
        _default_config = replace(_default_config, **kwargs)
    return get_config()


@contextmanager
def config_context(**kwargs: Any):
    """Temporarily override configuration for the current context.

    The override applies to this thread (and to asyncio tasks started from it)
    and nowhere else. The previous configuration is restored even if the body
    raises.

    Examples
    --------
    >>> import mlxlearn
    >>> with mlxlearn.config_context(fallback_policy="raise"):
    ...     mlxlearn.get_config().fallback_policy
    'raise'
    >>> mlxlearn.get_config().fallback_policy
    'warn'
    """
    _check_config_names(kwargs)
    outer = _config_scope.get()
    merged = {**outer.overrides, **kwargs} if outer is not None else dict(kwargs)
    # Built, and therefore validated, before anything changes.
    scope = _ConfigScope(merged, _default_config)
    token = _config_scope.set(scope)
    try:
        yield get_config()
    finally:
        _config_scope.reset(token)


def get_tuning() -> Tuning:
    """Return the private tuning values in effect here."""
    scope = _tuning_scope.get()
    return _default_tuning if scope is None else scope


def _process_tuning() -> Tuning:
    """The process-wide tuning default, ignoring any :func:`tuning_context`.

    For the few knobs that size something shared by every thread.
    """
    return _default_tuning


@contextmanager
def tuning_context(**kwargs: Any):
    """Temporarily override private tuning in the current context. Tests and debugging only."""
    known = {f.name for f in fields(Tuning)}
    unknown = set(kwargs) - known
    if unknown:
        raise TypeError(f"unknown tuning option(s): {', '.join(sorted(unknown))}")
    token = _tuning_scope.set(replace(get_tuning(), **kwargs))
    try:
        yield get_tuning()
    finally:
        _tuning_scope.reset(token)


def reset_config() -> None:
    """Restore configuration and tuning to their defaults.

    Resets the process-wide defaults to what the ``MLXLEARN_*`` environment says
    (re-reading it), and drops any override active in the current context.
    """
    global _default_config, _default_tuning
    config = _config_from_env()
    tuning = _tuning_from_env(Tuning())
    with _lock:
        _default_config = config
        _default_tuning = tuning
    _config_scope.set(None)
    _tuning_scope.set(None)


# Importable so a test can assert the public surface has not silently grown. The
# plan fixes this list at six entries; the 50 environment variables of the
# ancestor do not get to become 50 public options by accident.
PUBLIC_CONFIG_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(Config))
