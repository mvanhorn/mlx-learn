# Sketch and weighted logistic investigation

The native GPU loop checks the first eight accepted iterations together, then
checks the ninth, then uses groups of four. Once a fit has reached 32 accepted
iterations it submits up to 16 iterations at a time, bounded by the remaining
iteration budget. This amortizes command-buffer submission on long fits; the
GPU still stops internally when its existing convergence condition is met.
The objective, gradients, convergence thresholds, and sketch construction are
unchanged by this scheduling change.

The tuning study compared ten cases with eight repeats per native variant. In
that panel, fitted coefficients and intercepts were bit-identical and iteration
counts and status matched the previous schedule. Three ill-conditioned weighted
20,000 × 1,024 inputs improved from 80.39 to 75.96 ms, 83.01 to 77.00 ms, and
75.68 to 72.58 ms. A 68-step unbalanced fit changed from 39.59 to 40.12 ms.
Short fits retain their original scheduling. These are separate from the final
end-to-end benchmark below.

## End-to-end comparison

On an M4 Max, each entry is the median of 12 complete fresh fits after three
warmups, with the six implementations run in rotating order. Scikit-learn uses
one BLAS thread. Host timings include input validation and preparation. Device
timings start with resident MLX arrays and exclude the separately recorded
upload. The frozen baseline uses the previous native binary and source-verified
sketch code. Times are milliseconds.

| Data, 20,000 rows | Previous | Current | Previous sketch | Current sketch | sklearn | Device sketch |
|---|---:|---:|---:|---:|---:|---:|
| 1,024 features | 7.265 | 7.198 | 8.512 | 8.819 | 12.865 | 10.359 |
| 4,096 features | 42.511 | 42.837 | 51.676 | 52.783 | 65.160 | 56.136 |
| Weighted, ill-conditioned, 1,024 | 79.938 | 75.665 | 79.790 | 79.058 | 196.502 | 84.364 |
| Shifted, 1,024 | 13.665 | 13.661 | 13.179 | 13.248 | 17.591 | 11.730 |
| Rank-deficient, 1,024 | 8.943 | 8.989 | 9.679 | 10.165 | 14.979 | 11.250 |
| Ill-conditioned, 4,096 | 403.628 | 403.256 | 372.802 | 373.603 | 577.169 | 394.712 |
| Weighted, 1,024 | 7.288 | 7.241 | 8.448 | 8.747 | 13.114 | 10.647 |
| Weighted, 4,096 | 57.397 | 57.606 | 64.918 | 65.452 | 80.649 | 69.664 |

The sketch is still slower on ordinary wide data. The large shift makes the
stock sklearn result fail the original-coordinate gradient check, so its time
is not an accuracy-equivalent comparison; quality for that row was checked
against a tightly converged centered sklearn reference instead. The other
quality checks evaluate the original inputs and weights in float64 outside the
timer. All measured host parameters matched the corresponding frozen host
parameters bit-for-bit; the audited gradients, objectives, predictions, and
rank-deficient null-space checks passed for the listed cases. These checks cover
the fixtures, not all possible datasets. The [raw results](../benchmarks/results/sketch-weighted-final-2026-10-27.json)
include each timing, warning, objective, gradient, device-upload observation,
and source integrity check. Run the panel with
`python -m benchmarks.sketch_weighted --device --repeats 12 --warmup 3`.

A [second panel](../benchmarks/results/sketch-weighted-final-float64-2026-10-27.json)
tested four wide and weighted cases with float64 host inputs. Those inputs were
exact widenings of the float32 fixtures, so the panel exercises conversion but
does not cover arbitrary float64-only values. All four quality audits passed.

The source-verified frozen binary is kept locally at
`bench/sketch-weighted-start/_metal_loop.abi3.so`; its sources are archived in
`benchmarks/reference/sketch-weighted-start-sources-2026-10-27.zip`. A new
checkout must rebuild a matching frozen extension before reproducing that exact
comparison. Both frozen archives and the binary are hashed before and after
the run.

The sketch itself remains unchanged. A one-gradient-pair shortcut passed some
initial cases but had a 4.52e-4 relative objective regression on a held-out
unbalanced weighted case despite satisfying its gradient criterion. Fusing all
four probes was not consistently faster. Fusing only the Hessian action kept
the probe directions but altered reduction order: on a held-out ill-conditioned
4,096-feature input the host objective was 1.251e-5 worse than the original
sketch, and some device fits exhausted their budget. Neither was promoted.
An MPS dense Gram/Cholesky preconditioner also cost more than the entire long
weighted fit, while a projected row-space metric failed a held-out objective
check. The archived experiments and all observed failures remain in
`benchmarks/results/` and `benchmarks/reference/`.

There are pre-existing limitations. On an unbalanced weighted 1,024-feature
fixture, the gradient stopping rule can declare convergence while the objective
is materially above a tighter sklearn reference. For three unbalanced seeds,
the unsketched host objective was 0.61%–2.69% higher at `tol=1e-4`. In a
single-fit tolerance sweep, `tol=1e-6` reduced the gap to 1.15e-6–3.51e-6,
but increased accepted iterations from 16–19 to 37–39. This experiment does
not establish a safe automatic tolerance rule. The compensated device check
also reports nonconvergence for one held-out weighted ill-conditioned seed at
the default 200-iteration budget; both the old and new schedules do so. Neither
the public gradient tolerance nor the device safeguard is a general guarantee
of objective parity.

The full suite passed: 803 tests and 43 warnings. The native scheduling tests
include iteration budgets immediately before and after the new batch boundary.
