"""Center and validate a borrowed design while preparing owned GPU storage.

Binary fits can fuse their initial gradient and diagonal into the centering
copy. These are computed from the actual centered values, avoiding cancellation
from subtracting raw squared moments for features with large offsets.
"""

from __future__ import annotations

import mlx.core as mx

__all__ = ["prepare_centered"]

_center_stats = mx.fast.metal_kernel(
    name="mlxlearn_logistic_center_stats",
    input_names=["X"],
    output_names=["parts", "flags"],
    source=r"""
uint t=thread_position_in_threadgroup.x,lane=t%32,sg=t/32;
uint cb=threadgroup_position_in_grid.x,j=cb*32+lane;
uint b=threadgroup_position_in_grid.y;
float anchor=j<D?X[j]*.25f:0.0f,acc=0.0f;
bool finite=true;
if(j<D){for(ulong row=ulong(b)*256+sg;row<min(ulong(N),(ulong(b)+1)*256);row+=4){
 float x=X[row*D+j];finite=finite && isfinite(x);
 acc+=(x*.25f-anchor)*(1.0f/N);
}}
threadgroup float sums[128];
threadgroup uint valid[4];
sums[t]=acc;
bool good=simd_all(finite);
if(lane==0)valid[sg]=uint(good);
threadgroup_barrier(mem_flags::mem_threadgroup);
if(sg==0 && j<D)parts[ulong(b)*D+j]=(sums[lane]+sums[lane+32])+(sums[lane+64]+sums[lane+96]);
if(t==0)flags[ulong(b)*COL_BLOCKS+cb]=valid[0]&valid[1]&valid[2]&valid[3];
""",
)

_center_copy = mx.fast.metal_kernel(
    name="mlxlearn_logistic_center_copy",
    input_names=["X", "center"],
    output_names=["owned"],
    source=r"""
uint t=thread_position_in_threadgroup.x;
ulong block=ulong(threadgroup_position_in_grid.x)+
    ulong(threadgroup_position_in_grid.y)*BLOCK_COLUMNS;
ulong size=ulong(N)*D;
for(uint k=0;k<4;++k){
 ulong i=block*1024+t+k*256;
 if(i<size)owned[i]=X[i]-center[i%D];
}
""",
)

_center_initial = mx.fast.metal_kernel(
    name="mlxlearn_logistic_center_initial",
    input_names=["X", "center", "y", "sw"],
    output_names=["owned", "parts"],
    source=r"""
// Match native_logistic._initial's row partition and accumulation order.
// Every feature value is copied, including rows with zero sample weight.
uint tid=thread_position_in_threadgroup.x;
uint col=threadgroup_position_in_grid.x*32+tid%32;
uint block=threadgroup_position_in_grid.y,sg=tid/32;
float grad=0,diag=0;
if(col<P){for(ulong row=ulong(block)*256+sg;row<min(ulong(N),(ulong(block)+1)*256);row+=4){
 float x=1.0f;
 if(col<D){ulong i=row*D+col;x=X[i]-center[col];owned[i]=x;}
 float weight=sw[row];
 if(weight>0){grad+=x*(.5f-y[row])*weight;diag+=.25f*x*x*weight;}
}}
threadgroup float gp[128],hp[128];gp[tid]=grad;hp[tid]=diag;
threadgroup_barrier(mem_flags::mem_threadgroup);
if(sg==0 && col<P){
 parts[ulong(block)*P+col]=gp[tid]+gp[tid+32]+gp[tid+64]+gp[tid+96];
 parts[ulong(B)*P+ulong(block)*P+col]=hp[tid]+hp[tid+32]+hp[tid+64]+hp[tid+96];
}
""",
)


@mx.compile
def _center_and_check(X):
    n, d = X.shape
    nb, cols = (n + 255) // 256, (d + 31) // 32
    parts, flags = _center_stats(
        inputs=[X],
        template=[("N", n), ("D", d), ("COL_BLOCKS", cols)],
        grid=(cols * 128, nb, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(nb, d), (nb, cols)],
        output_dtypes=[mx.float32, mx.uint32],
    )
    # Quarter-scaled differences cannot exceed half the float32 range. Clipping
    # the rounded mean before restoring its scale also protects finite constants.
    bound = 3.4028234663852886e38 * 0.25
    center = mx.clip(X[0] * 0.25 + mx.sum(parts, axis=0), -bound, bound) * 4.0
    return center, mx.all(flags != 0)


def prepare_centered(X, y=None, sw=None):
    """Return lazy owned data, its applied center, finiteness, and initial parts.

    ``X`` must be a nonempty float32 MLX matrix. When ``y`` is supplied, it
    contains binary labels and the returned unnormalized initial parts have
    shape ``(2, ceil(n / 256), d + 1)``; the last column is the intercept.
    Without labels, the initial-parts result is ``None``.

    The caller must evaluate every returned array before releasing or modifying
    borrowed input storage. Finiteness describes the original input, not the
    centered differences, matching ordinary input validation.
    """
    if X.ndim != 2 or min(X.shape) == 0 or X.dtype != mx.float32:
        raise ValueError("prepare_centered requires a nonempty float32 matrix")
    n, d = X.shape
    center, finite = _center_and_check(X)
    if y is None:
        blocks = (X.size + 1023) // 1024
        columns = min(blocks, 65535)
        owned = _center_copy(
            inputs=[X, center],
            template=[("N", n), ("D", d), ("BLOCK_COLUMNS", columns)],
            grid=(columns * 256, (blocks + columns - 1) // columns, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[X.shape],
            output_dtypes=[mx.float32],
        )[0]
        return owned, center, finite, None
    if y.shape != (n,) or (sw is not None and sw.shape != (n,)):
        raise ValueError("labels and sample weights must match the design's rows")
    if sw is None:
        sw = mx.ones(n, dtype=X.dtype)
    p, nb = d + 1, (n + 255) // 256
    owned, parts = _center_initial(
        inputs=[X, center, y, sw],
        template=[("N", n), ("D", d), ("P", p), ("B", nb)],
        grid=(((p + 31) // 32) * 128, nb, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[X.shape, (2, nb, p)],
        output_dtypes=[mx.float32, mx.float32],
    )
    return owned, center, finite, parts
