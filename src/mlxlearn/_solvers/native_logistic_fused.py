"""One matrix pass for line search and its first candidate's gradient.

The selected candidate is still decided by the ordinary Armijo kernel. This
module computes the gradient for the first candidate speculatively; callers may
reuse it only when that candidate was actually accepted. Other accepted steps
retain the normal gradient evaluation, and final convergence is checked directly.

Each SIMD group retains its feature slice in registers and accumulates across
several rows before writing partial gradients. Wider rows span four SIMD groups
to bound register pressure. The shared workspace is reused for row dot products,
losses/intercepts, and feature gradients, never exceeding 32 KiB.
"""

from __future__ import annotations

import mlx.core as mx

# Imported only when native_logistic.step first needs the wide kernel. That
# module is fully initialized before this module is loaded.
from .native_logistic import _REDUCE

__all__ = ["forward_gradient"]

_speculative = mx.fast.metal_kernel(
    name="mlxlearn_native_lbfgs_speculative",
    header=_REDUCE,
    input_names=["X", "direction", "raw", "y", "sw", "stats", "denom"],
    output_names=["dr", "partial", "gradient_parts"],
    source=r"""

uint tid=thread_position_in_threadgroup.x,lane=tid%32,sg=tid/32;
uint group=threadgroup_position_in_grid.x;
constexpr uint V=(D+32*FEATURE_GROUPS-1)/(32*FEATURE_GROUPS);
constexpr uint CHUNK=V*32,DPAD=CHUNK*FEATURE_GROUPS;
constexpr uint ROW_GROUPS=GROUPS/FEATURE_GROUPS;
constexpr uint WS_GRAD=ROW_GROUPS*DPAD,WS_LOSS=GROUPS*(CAND+1);
constexpr uint SCRATCH=WS_GRAD>WS_LOSS?WS_GRAD:WS_LOSS;
uint fs=sg%FEATURE_GROUPS,rs=sg/FEATURE_GROUPS;
threadgroup float scratch[SCRATCH];
float accumulator[V];for(uint k=0;k<V;++k)accumulator[k]=0;
float local_loss=0,local_bias=0;
float alpha0=stats[8];bool active=stats[1]==0;
for(uint rr=0;rr<ROWS;++rr){
 uint row=group*(ROW_GROUPS*ROWS)+rr*ROW_GROUPS+rs;
 float values[V],z=0;
 if(row<N && active){for(uint k=0;k<V;++k){uint j=fs*CHUNK+lane+k*32;values[k]=j<D?X[ulong(row)*D+j]:0.0f;if(j<D)z=fma(values[k],direction[j],z);}}
 else{for(uint k=0;k<V;++k)values[k]=0;}
 z=simd_sum(z);
 if(FEATURE_GROUPS>1){
  if(lane==0)scratch[sg]=z;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  z=0;for(uint f=0;f<FEATURE_GROUPS;++f)z+=scratch[rs*FEATURE_GROUPS+f];
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 if(P>D && active)z+=direction[D];
 if(lane==0 && fs==0 && row<N)dr[row]=z;
 float first_residual=0;
 if(row<N && active && sw[row]>0){
  float sign=1-2*y[row],u=sign*raw[row];
  if(fs==0 && lane<CAND){
   float alpha=alpha0*exp2(-float(lane)),dz=sign*z*alpha;
   float diff=abs(dz)<.5f?log1p(sigmoid_stable(u)*expm1_small(dz)):softplus_stable(u+dz)-softplus_stable(u);
   local_loss+=sw[row]*diff/denom;
  }
  if(lane==0){first_residual=sw[row]*sign*sigmoid_stable(sign*fma(alpha0,z,raw[row]));if(fs==0)local_bias+=first_residual;}
 }
 first_residual=simd_shuffle(first_residual,ushort(0));
 for(uint k=0;k<V;++k)accumulator[k]=fma(values[k],first_residual,accumulator[k]);
}
if(lane<CAND)scratch[sg*(CAND+1)+lane]=local_loss;
if(lane==0)scratch[sg*(CAND+1)+CAND]=local_bias;
threadgroup_barrier(mem_flags::mem_threadgroup);
if(sg==0){
 if(lane<CAND){float total=0;for(uint s=0;s<GROUPS;++s)total+=scratch[s*(CAND+1)+lane];partial[ulong(lane)*B+group]=total;}
 if(lane==0 && P>D){float total=0;for(uint s=0;s<GROUPS;++s)total+=scratch[s*(CAND+1)+CAND];gradient_parts[ulong(group)*P+D]=total;}
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for(uint k=0;k<V;++k){uint j=fs*CHUNK+lane+k*32;scratch[rs*DPAD+j]=accumulator[k];}
threadgroup_barrier(mem_flags::mem_threadgroup);
if(rs==0){for(uint k=0;k<V;++k){uint j=fs*CHUNK+lane+k*32;if(j<D){float total=0;for(uint r=0;r<ROW_GROUPS;++r)total+=scratch[r*DPAD+j];gradient_parts[ulong(group)*P+j]=total;}}}

""",
)


@mx.compile
def forward_gradient(X, direction, raw, y, sw, stats, denom, candidates):
    """Return projections, candidate loss changes, and the first sample gradient.

    The gradient is normalized by ``denom`` and excludes regularization.
    ``stats[8]`` supplies the initial step, including a resumed line search.
    All inputs are explicit; no training buffer is retained between fits.
    """
    n, d = X.shape
    p = direction.size
    if not 512 <= d <= 4096:
        raise ValueError("wide native fusion requires 512 to 4096 features")
    if not 1 <= candidates <= 32:
        raise ValueError("wide native fusion requires 1 to 32 candidates")
    if d <= 1024:
        groups, feature_groups, rows = 2, 1, 16
    else:
        groups, feature_groups, rows = 8, 4, 8
    if n >= 16384:
        # Longer row batches reduce partial-gradient traffic on large designs.
        # Keep smaller problems' launch occupancy; the split-row geometry still
        # bounds registers and shared storage independently of the row count.
        if d <= 1024:
            groups, rows = 4, 32
        else:
            rows = 32 if d <= 2048 else 64
    rows_per_group = (groups // feature_groups) * rows
    blocks = (n + rows_per_group - 1) // rows_per_group
    dr, partial, gradient_parts = _speculative(
        inputs=[X, direction, raw, y, sw, stats, denom],
        template=[
            ("N", n),
            ("D", d),
            ("P", p),
            ("B", blocks),
            ("GROUPS", groups),
            ("ROWS", rows),
            ("FEATURE_GROUPS", feature_groups),
            ("CAND", candidates),
        ],
        grid=(blocks * groups * 32, 1, 1),
        threadgroup=(groups * 32, 1, 1),
        output_shapes=[(n,), (candidates, blocks), (blocks, p)],
        output_dtypes=[mx.float32] * 3,
    )
    return dr, mx.sum(partial, axis=1), mx.sum(gradient_parts, axis=0) / denom
