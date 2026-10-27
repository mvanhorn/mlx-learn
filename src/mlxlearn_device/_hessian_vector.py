"""Private, NumPy-free Hessian-vector products for binary logistic solvers.

The Hessian applies to the actual centered float32 design. Per-row curvature
includes sample weights and normalization; regularization is supplied as a
vector so an intercept can remain unpenalized. Callers own and validate the
design for the complete asynchronous operation and must reject nonfinite
outputs. This module is not wired into the public optimizer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx

_one_pass_hvp = mx.fast.metal_kernel(
    name="mlxlearn_device_single_pass_hvp",
    input_names=["X", "curvature", "vector"],
    output_names=["parts"],
    source=r"""
uint tid=thread_position_in_threadgroup.x,lane=tid%32,sg=tid/32;
uint block=threadgroup_position_in_grid.x;
constexpr uint V=(D+32*FEATURE_GROUPS-1)/(32*FEATURE_GROUPS);
constexpr uint CHUNK=V*32,DPAD=CHUNK*FEATURE_GROUPS;
constexpr uint ROW_GROUPS=GROUPS/FEATURE_GROUPS;
uint fs=sg%FEATURE_GROUPS,rs=sg/FEATURE_GROUPS;
threadgroup float scratch[ROW_GROUPS*DPAD];
float accumulator[V];for(uint k=0;k<V;++k)accumulator[k]=0.0f;
float local_bias=0.0f;
for(uint rr=0;rr<ROWS;++rr){
 uint row=block*(ROW_GROUPS*ROWS)+rr*ROW_GROUPS+rs;
 float c=row<N?curvature[row]:0.0f;
 bool invalid=row<N && (!isfinite(c) || c<0.0f);
 bool load=row<N && (c>0.0f || invalid);
 float values[V],dot=0.0f;
 if(load){for(uint k=0;k<V;++k){uint j=fs*CHUNK+lane+k*32;
  values[k]=j<D?X[ulong(row)*D+j]:0.0f;
  if(j<D)dot=fma(values[k],vector[j],dot);
 }}else{for(uint k=0;k<V;++k)values[k]=0.0f;}
 dot=simd_sum(dot);
 if(FEATURE_GROUPS>1){
  if(lane==0)scratch[sg]=dot;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  dot=0.0f;for(uint f=0;f<FEATURE_GROUPS;++f)dot+=scratch[rs*FEATURE_GROUPS+f];
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 if(P>D)dot+=vector[D];
 float weighted=0.0f;
 if(lane==0 && row<N){
  weighted=invalid?as_type<float>(0x7fc00000u):(c==0.0f?0.0f:c*dot);
  if(fs==0)local_bias+=weighted;
 }
 weighted=simd_shuffle(weighted,ushort(0));
 for(uint k=0;k<V;++k)accumulator[k]=fma(values[k],weighted,accumulator[k]);
}
if(lane==0)scratch[sg]=local_bias;
threadgroup_barrier(mem_flags::mem_threadgroup);
if(sg==0 && lane==0 && P>D){float total=0.0f;
 for(uint s=0;s<GROUPS;++s)total+=scratch[s];
 parts[ulong(block)*P+D]=total;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for(uint k=0;k<V;++k){uint j=fs*CHUNK+lane+k*32;scratch[rs*DPAD+j]=accumulator[k];}
threadgroup_barrier(mem_flags::mem_threadgroup);
if(rs==0){for(uint k=0;k<V;++k){uint j=fs*CHUNK+lane+k*32;if(j<D){
 float total=0.0f;for(uint r=0;r<ROW_GROUPS;++r)total+=scratch[r*DPAD+j];
 parts[ulong(block)*P+j]=total;
}}}
""",
)


def _geometry(n: int, d: int):
    if d <= 1024:
        groups, feature_groups, rows = 2, 1, 16
    else:
        groups, feature_groups, rows = 8, 4, 8
    if n >= 16384:
        if d <= 1024:
            groups, rows = 4, 32
        else:
            rows = 32 if d <= 2048 else 64
    blocks = (n + (groups // feature_groups) * rows - 1) // ((groups // feature_groups) * rows)
    return groups, feature_groups, rows, blocks


def _fused_hvp(X, curvature, vector, regularization):
    """Compute an exact-form HVP in float32; final finite checks remain required."""
    if X.ndim != 2 or not X.shape[0] or not 1 <= X.shape[1] <= 4096:
        raise ValueError("X must be a nonempty matrix of at most 4096 features")
    n, d = X.shape
    if (X.dtype != mx.float32 or curvature.dtype != mx.float32 or vector.dtype != mx.float32
        or regularization.dtype != mx.float32 or curvature.shape != (n,)
        or vector.shape not in ((d,), (d + 1,)) or regularization.shape != vector.shape):
        raise ValueError("curvature, vector and regularization must match float32 design")
    p = vector.shape[0]
    groups, feature_groups, rows, blocks = _geometry(n, d)
    parts = _one_pass_hvp(
        inputs=[X, curvature, vector],
        template=[("N", n), ("D", d), ("P", p), ("B", blocks), ("GROUPS", groups),
                  ("FEATURE_GROUPS", feature_groups), ("ROWS", rows)],
        grid=(blocks * groups * 32, 1, 1), threadgroup=(groups * 32, 1, 1),
        output_shapes=[(blocks, p)], output_dtypes=[mx.float32],
    )[0]
    return mx.sum(parts, axis=0) + regularization * vector


_compiled_fused_hvp = mx.compile(_fused_hvp)


@dataclass(frozen=True)
class MetalLimits:
    """Verified limits for this exact compiled HVP specialization.

    The device limit alone is insufficient for a register-heavy kernel. The
    compiled pipeline's ``maxTotalThreadsPerThreadgroup`` and static memory
    allocation must be supplied separately for this exact specialization.
    Missing pipeline capabilities always select the two-pass fallback.
    """

    available: bool
    device_max_threads_per_group: int
    max_threadgroup_bytes: int
    max_buffer_bytes: int
    pipeline_max_threads_per_group: int | None = None
    pipeline_static_threadgroup_bytes: int | None = None
    max_workspace_bytes: int = 64 * 1024 * 1024


@dataclass(frozen=True)
class _PreparedHessianDesign:
    """One row-contiguous design and the device selected for this fit."""

    array: mx.array
    device: object


def hessian_vector_plan(n: int, d: int, p: int, limits: MetalLimits | None):
    """Select a bounded Metal launch or explain why a two-pass path is needed."""
    if n <= 0 or d <= 0 or p not in (d, d + 1):
        raise ValueError("nonempty matrix and compatible parameter vector required")
    if limits is None or limits.available is False:
        return {"fused": False, "reason": "metal-capabilities-unavailable"}
    if limits.available is not True:
        return {"fused": False, "reason": "invalid-metal-capabilities"}
    if limits.pipeline_max_threads_per_group is None or limits.pipeline_static_threadgroup_bytes is None:
        return {"fused": False, "reason": "pipeline-capabilities-unavailable"}
    capacities = (limits.device_max_threads_per_group, limits.max_threadgroup_bytes,
                  limits.max_buffer_bytes, limits.pipeline_max_threads_per_group,
                  limits.pipeline_static_threadgroup_bytes, limits.max_workspace_bytes)
    if any(type(value) is not int or value <= 0 for value in capacities):
        return {"fused": False, "reason": "invalid-metal-capabilities"}
    if d > 4096 or n > 2**31 - 1:
        return {"fused": False, "reason": "unsupported-dimensions"}
    groups, feature_groups, rows, blocks = _geometry(n, d)
    row_groups = groups // feature_groups
    padded = ((d + 32 * feature_groups - 1) // (32 * feature_groups)) * 32 * feature_groups
    scratch_bytes = row_groups * padded * 4
    partial_bytes = blocks * p * 4
    grid_threads = blocks * groups * 32
    if groups * 32 > min(limits.device_max_threads_per_group, limits.pipeline_max_threads_per_group):
        return {"fused": False, "reason": "threadgroup-size-limit"}
    if (limits.pipeline_static_threadgroup_bytes < scratch_bytes
        or limits.pipeline_static_threadgroup_bytes > limits.max_threadgroup_bytes
        or scratch_bytes > min(32 * 1024, limits.max_threadgroup_bytes)):
        return {"fused": False, "reason": "threadgroup-memory-limit"}
    if partial_bytes > min(limits.max_workspace_bytes, limits.max_buffer_bytes):
        return {"fused": False, "reason": "workspace-limit"}
    if grid_threads > 2**31 - 1:
        return {"fused": False, "reason": "grid-limit"}
    return {"fused": True, "groups": groups, "feature_groups": feature_groups,
            "rows": rows, "blocks": blocks, "scratch_bytes": scratch_bytes,
            "partial_bytes": partial_bytes, "grid_threads": grid_threads}


def prepare_hessian_design(X, *, device):
    """Prepare one row-contiguous design outside the inner CG loop.

    The caller must evaluate the returned array before releasing borrowed
    input storage, refrain from mutating that storage, and retain the wrapper
    for all asynchronous operations. The array may share the input buffer.
    Finiteness must be validated by the caller's input-preparation step.
    """
    if not isinstance(X, mx.array) or X.ndim != 2 or X.dtype != mx.float32 or min(X.shape) == 0:
        raise ValueError("X must be a nonempty float32 MLX matrix")
    if device not in (mx.cpu, mx.gpu):
        raise ValueError("device must select the MLX CPU or GPU")
    if device == mx.gpu and not mx.metal.is_available():
        raise RuntimeError("the requested MLX Metal device is unavailable")
    return _PreparedHessianDesign(mx.contiguous(X, stream=device), device)


def logistic_curvature(logits, sample_weight, denominator, *, device):
    """Stable represented curvature; invalid active rows propagate NaNs.

    A zero-weight row contributes exactly zero, even if its unused logit
    overflowed. Active rows require finite logits and nonnegative finite
    weights. Saturated positive logits retain the representable contribution
    from sigmoid(-z), which ``1 - sigmoid(z)`` would round away.
    """
    if (not isinstance(logits, mx.array) or not isinstance(sample_weight, mx.array)
        or logits.ndim != 1 or sample_weight.shape != logits.shape
        or logits.dtype != mx.float32 or sample_weight.dtype != mx.float32):
        raise ValueError("logits and sample weights must be matching float32 MLX vectors")
    if isinstance(denominator, bool) or not isinstance(denominator, (float, int)) or not math.isfinite(denominator):
        raise ValueError("denominator must be finite and positive")
    if not 1.1754943508222875e-38 <= denominator <= 3.4028234663852886e38:
        raise ValueError("denominator must be a normal positive float32 value")
    with mx.stream(device):
        zero = sample_weight == 0
        valid = (mx.isfinite(sample_weight) & (sample_weight >= 0)
                 & (zero | mx.isfinite(logits)))
        safe_logits = mx.where(zero, mx.zeros_like(logits), logits)
        curvature = mx.sigmoid(safe_logits) * mx.sigmoid(-safe_logits) * sample_weight / denominator
        curvature = mx.where(zero, mx.zeros_like(curvature), curvature)
        return mx.where(valid, curvature, mx.full_like(curvature, float("nan")))


def _two_pass_hvp(X, curvature, vector, regularization, *, device):
    """Safe same-device fallback for finite, caller-validated design inputs."""
    with mx.stream(device):
        d = X.shape[1]
        projection = X @ vector[:d]
        if vector.shape[0] == d + 1:
            projection = projection + vector[d]
        # A row with exactly zero curvature can overflow in Xv; select zero
        # before the backward multiply. X itself is validated finite upstream.
        weighted = mx.where(curvature == 0, mx.zeros_like(curvature), curvature * projection)
        result = X.T @ weighted
        if vector.shape[0] == d + 1:
            result = mx.concatenate([result, mx.sum(weighted, keepdims=True)])
        result = result + regularization * vector
        valid = mx.all(mx.isfinite(curvature) & (curvature >= 0))
        return mx.where(valid, result, mx.full_like(result, float("nan")))


def hessian_vector_product(design, curvature, vector, regularization, *, device=None, limits=None):
    """Return a lazy HVP and launch plan without importing NumPy or sklearn.

    The prepared design must be validated finite and held alive by the caller.
    Requiring the wrapper prevents accidentally recopying a non-contiguous
    design on each inner iteration. Rows with zero curvature are skipped before
    the Metal dot product. Invalid curvature propagates a nonfinite result;
    callers must reject that result and run an independent final certificate.
    """
    if not isinstance(design, _PreparedHessianDesign):
        raise ValueError("design must be prepared once by prepare_hessian_design")
    if device is not None and device != design.device:
        raise ValueError("device must match the prepared design")
    device = design.device
    X = design.array
    if not all(isinstance(value, mx.array) for value in (curvature, vector, regularization)):
        raise ValueError("curvature, vector and regularization must be MLX arrays")
    n, d = X.shape
    if (curvature.dtype != mx.float32 or vector.dtype != mx.float32 or regularization.dtype != mx.float32
        or curvature.shape != (n,) or vector.shape not in ((d,), (d + 1,))
        or regularization.shape != vector.shape):
        raise ValueError("curvature, vector and regularization must match the design")
    chosen = hessian_vector_plan(n, d, vector.shape[0], limits if device == mx.gpu else None)
    with mx.stream(device):
        if chosen["fused"]:
            return _compiled_fused_hvp(X, curvature, vector, regularization), chosen
        return _two_pass_hvp(X, curvature, vector, regularization, device=device), chosen
