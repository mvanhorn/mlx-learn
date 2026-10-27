# Frozen logistic benchmark references

`native-logistic-2026-10-26.zip` contains the 33 package source files behind the
preceding native logistic regression benchmark. Its solver, estimator,
validation, and configuration hashes match
[`native-logistic-2026-10-26.json`](../results/native-logistic-2026-10-26.json).
There are no training inputs, fitted parameters, bytecode, or compiled binaries
in this archive. The source retains the repository's [Apache-2.0 license](../../LICENSE).

The companion manifest records each source hash and the complete archive hash.
The wide benchmark uses this archive by default, including on a fresh checkout:

```bash
python -m benchmarks.logistic_wide --profile full --dtype both
```

`wide-logistic-2026-10-26.zip` contains the 35 package source files behind
[`wide-logistic-2026-10-26.json`](../results/wide-logistic-2026-10-26.json),
including fused wide kernels and GPU input validation. Every source hash and the
complete source inventory were verified against that result before capturing the
archive. Its manifest also records the published result's hash. Use this newer
reference to measure subsequent conversion and scheduling changes:

```bash
python -m benchmarks.logistic_wide --profile full --dtype both \
    --reference-archive benchmarks/reference/wide-logistic-2026-10-26.zip \
    --repeats 15 --warmup 3
python -m benchmarks.logistic_profile --dtype both \
    --reference-archive benchmarks/reference/wide-logistic-2026-10-26.zip
```

Before importing the reference under an isolated package name, the runner checks
the archive hash, exact entry set, canonical paths, regular-file types, allowed
source extensions, and individual file hashes. It writes only verified entries
to a temporary directory, which is removed when the run ends. It does not use
unrestricted ZIP extraction.

During reference-module imports, the runner temporarily wraps MLX's Metal-kernel
factory to prefix frozen kernel symbols with `wide_reference_<source-hash>_`. It restores the
factory before any fit. This separates old and new kernel identities without
changing the verified source files or numerical expressions, and avoids relying
on undocumented compilation-cache keys. All source modules that declare Metal
kernels are eagerly imported inside this namespace, including helpers that would
otherwise load lazily during a fit. No kernel is evaluated while importing. The
prefix and module/kernel names are recorded in the results.

A local `--reference-dir` override can select a directory containing `mlxlearn/`
and `manifest.json`. An alternative `--reference-archive` must have an adjacent
manifest with the same basename and `.manifest.json` extension.

Keep both references immutable. Future optimizer changes belong in the live
package; replacing a reference would change the meaning of the before/after
comparison.

`overhead-logistic-2026-10-26.zip` captures the complete package after the
conversion and GPU preparation optimizations. Its 37 files match the package
source inventory and hashes recorded in
[`overhead-logistic-2026-10-26-final.json`](../results/overhead-logistic-2026-10-26-final.json).
The adjacent manifest records both the benchmark-result hash and the archive
hash. Use it as the frozen baseline for native command-loop experiments:

```bash
python -m benchmarks.logistic_wide --profile full --dtype both \
    --reference-archive benchmarks/reference/overhead-logistic-2026-10-26.zip \
    --repeats 30 --warmup 3 --threads 1
```
