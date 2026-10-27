# Native Metal command-loop prototype — 2026-10-26

The [prototype archive](../benchmarks/reference/metal-command-prototype-2026-10-26.zip)
contains the full package source, Objective-C++ extension source, build configuration,
and the exploratory probe scripts and logs. The adjacent [manifest](../benchmarks/reference/metal-command-prototype-2026-10-26.manifest.json)
records SHA-256 hashes, including the hash of the binary observed during the probes.
Compiled binaries are not included; rebuild the extension for the target interpreter
and machine. The archive does not assert that this prototype is the shipped solver.

The captured best version uses one per-fit aligned Metal scratch allocation,
four-way partial reduction, and batches up to eight optimizer steps per command
buffer. It consumes evaluated MLX Metal DLPack buffers, retains their owners until
synchronous Metal completion, and checks the result with the existing direct
objective and gradient. A rejected first search batch, a required reanchor, or
failure to certify convergence falls back by restarting the established solver.
That fallback preserves the established result contract but can repeat work.

The source and scripts correspond to Apple M4 Max, Python 3.13.3, MLX 0.32.2,
NumPy 2.5.2 and scikit-learn 1.9.0. A Metal-capable Apple Silicon Mac, matching
dependencies, and an Objective-C++ compiler with the macOS SDK are required for
building the extension. Runtime shader compilation uses the Metal system API.
The probe scripts use synthetic binary classification inputs, seed 41, and one
requested thread for threadpoolctl-recognized pools. The smaller probe warms up
four times and the larger probe three; both report all remaining samples and
alternate the implementation order.

```bash
experiment_dir=$(mktemp -d)
unzip -q benchmarks/reference/metal-command-prototype-2026-10-26.zip -d "$experiment_dir"
cd "$experiment_dir"
python3 setup.py build_ext --inplace --force
PYTHONPATH="$PWD/src" python3 bench/native_loop/compare.py
PYTHONPATH="$PWD/src" python3 bench/native_loop/compare-large.py
```

The final exploratory 20k × 1,024 float32 run measured median **6.97 ms** for the
command loop, **7.16 ms** for the previous MLX loop, and **13.12 ms** for sklearn
over 12 samples each. The 60k × 1,024 probe measured **14.04 ms**, **13.36 ms**,
and **35.65 ms**, respectively, over nine samples each. These are exploratory
measurements with a broad spread and a shared machine; they do not demonstrate a
reliable complete-fit speedup. Both versions agreed with sklearn on 99.965% of
the 20k predictions and 99.957% of the 60k predictions in the last fitted model.
The full benchmark's independent objective and gradient checks are stricter than
these exploratory prediction comparisons.

The captured per-fit buffer-allocation and eight-step variants, raw logs, and a
small stage probe are included for inspection. Earlier individual-allocation and
serial-reduction timings are exploratory records; the archive does not contain
every intermediate source variant needed to recreate each one exactly. The final
source includes CPU-visible initial-gradient access, explicit producer and
consumer synchronization, bounded pipeline caching, and GPU memory barriers.
Metal command-buffer and encoder allocation failure handling requires additional
review before treating the prototype as production-ready.
