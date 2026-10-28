# GPU and subinterpreter CI compatibility

The original macOS 15 CI run exposed Metal loss-difference rounding and a
Python 3.12 isolated-subinterpreter import failure. Small native loss changes
now use a bounded explicit `log1p` series that retains the change under both
fast and safe Metal math. Weighted native fits use a 5% internal tolerance
reserve; the independent final gradient and progress checks, public tolerance,
and iteration budget remain unchanged. The native extension uses multi-phase
initialization with per-module ownership registries and retains `None`
explicitly for the Python 3.10 stable ABI.

Weighted, shifted multiclass fits have a separate intercept-refinement path.
It uses stable small-step exponential and logarithm series and normalizes
softmax probabilities directly, preserving the existing float64 objective and
original-coordinate gradient certificate. Independent tests cover represented
steps from `1e-9` to `0.45`, zero weights and reduction-block boundaries.

The Python 3.13-built extension passed sequential import, lifetime, and
reference-count checks on Python 3.10–3.14. Concurrent isolated-interpreter
checks passed on 3.12–3.14. A legacy shared-GIL Python 3.11 concurrent import
control can hang even without the extension, so that separate stress test is
limited to Python versions with per-interpreter GIL support.

On the M4 Max with MLX 0.32.4 and NumPy 2.5.4, the full suite passed **1,177
tests**, with 43 warnings. An independent float64 objective, gradient and
prediction audit passed for each fit in the following supplemental panel. It
uses the existing fixtures and frozen baseline, one warmup and two measured
fits; it is a correctness and performance sanity check, not a precise new
performance ranking. The harness requests a thread-pool limit of one, but
`threadpoolctl` does not expose Apple Accelerate's BLAS pool on this machine.

| 20,000-row dataset | Frozen ms | Current ms | sklearn ms | Current max gradient |
| --- | ---: | ---: | ---: | ---: |
| Wide, 4,096 features | 58.100 | 29.130 | 67.730 | 5.72e-6 |
| Weighted ill-conditioned, 1,024 features | 75.569 | 75.776 | 211.961 | 3.12e-5 |
| Shifted, 1,024 features | 13.265 | 13.502 | 17.715 | 4.51e-5 |
| Rank-deficient, 1,024 features | 9.332 | 9.258 | 15.054 | 6.77e-5 |
| Ill-conditioned, 4,096 features | 400.423 | 251.993 | 545.575 | 7.90e-7 |

For shifted inputs the stock sklearn fit did not meet the original-coordinate
gradient threshold; the audit used a separately verified centered reference.
All current host, sketch and resident-device variants passed the panel's
quality checks, and source hashes matched before and after measurement. The
[raw panel](../benchmarks/results/ci-compatibility-2026-10-27.json) contains
all samples, references, warnings and scores. Hosted CI is the authoritative
check for the macOS 15 GPU and compatibility matrix.
