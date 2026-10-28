"""Compensated Metal scoring of a binary model on original float32 inputs.

Two-float accumulators reduce cancellation caused by large feature offsets.
The transcendental approximation and arithmetic do not provide a formal error
bound; callers must treat the output as a numerical check, not a proof.
"""

from __future__ import annotations

import mlx.core as mx

_HEADER = r"""
inline float2 rn(float x,float y){float h=x+y;return float2(h,y-(h-x));}
inline float2 add2(float2 a,float2 b){float s=a.x+b.x;float v=s-a.x;float e=(a.x-(s-v))+(b.x-v)+a.y+b.y;return rn(s,e);}
inline float2 neg2(float2 a){return -a;}
inline float2 mul2(float2 a,float2 b){float p=a.x*b.x;float e=fma(a.x,b.x,-p)+a.x*b.y+a.y*b.x;return rn(p,e);}
inline float2 div2(float2 a,float2 b){float q=a.x/b.x;float2 r=add2(a,neg2(mul2(b,float2(q,0))));return add2(float2(q,0),float2((r.x+r.y)/b.x,0));}
inline float2 sum2(float2 a){for(uint k=16;k>0;k>>=1){float2 b=float2(simd_shuffle_down(a.x,k),simd_shuffle_down(a.y,k));a=add2(a,b);}return a;}
inline float2 exp2x(float2 z){
 if(z.x < -80.0f)return float2(0.0f,0.0f);
 float kf=rint(z.x*1.4426950408889634f); int k=int(kf);
 float2 r=add2(z,neg2(mul2(float2(kf,0),float2(0.6931471824645996f,-1.904654323148236e-9f))));
 float2 p=float2(2.08767569878681e-9f,0);
 p=add2(float2(2.505210838544172e-8f,0),mul2(r,p));
 p=add2(float2(2.755731998149713e-7f,0),mul2(r,p));
 p=add2(float2(2.7557319223985893e-6f,0),mul2(r,p));
 p=add2(float2(2.4801587642286904e-5f,0),mul2(r,p));
 p=add2(float2(1.9841270113829523e-4f,0),mul2(r,p));
 p=add2(float2(1.3888889225199819e-3f,0),mul2(r,p));
 p=add2(float2(8.333333767950535e-3f,0),mul2(r,p));
 p=add2(float2(4.16666679084301e-2f,0),mul2(r,p));
 p=add2(float2(1.666666716337204e-1f,0),mul2(r,p));
 p=add2(float2(0.5f,0),mul2(r,p));
 p=add2(float2(1.0f,0),mul2(r,p));
 p=add2(float2(1.0f,0),mul2(r,p));
 return float2(ldexp(p.x,k),ldexp(p.y,k));
}
"""

_forward = mx.fast.metal_kernel(
    name="mlxlearn_device_logistic_compensated_forward",
    header=_HEADER,
    input_names=["X", "center", "theta", "y", "sw", "denom"],
    output_names=["resid_hi", "resid_lo", "loss", "tail_error"],
    source=r"""
uint t=thread_position_in_threadgroup.x,lane=t%32,sg=t/32;
uint row=threadgroup_position_in_grid.x*4+sg;
float2 dot=float2(0);float local_max=1.0f;
bool active=row<N && sw[row]>0.0f;
if(active){for(uint j=lane;j<D;j+=32){
 float x=X[ulong(row)*D+j];local_max=max(local_max,abs(x));
 float2 dx=add2(float2(x,0),float2(-center[j],0));
 dot=add2(dot,mul2(dx,float2(theta[j],0)));
}}
dot=sum2(dot);float feature_max=simd_max(local_max);
if(lane==0 && row<N){
 if(!active){resid_hi[row]=0.0f;resid_lo[row]=0.0f;loss[row]=0.0f;tail_error[row]=0.0f;}
 else{
 dot=add2(dot,float2(theta[D],0));float sign=1.0f-2.0f*y[row];float2 u=sign*dot;
 float2 z=(u.x>=0 ? neg2(u):u);float2 e=exp2x(z);
 float2 probability=div2(u.x>=0 ? float2(1,0):e,add2(float2(1,0),e));
 float2 weight=div2(float2(sw[row],0),float2(denom[0],denom[1]));
 float2 r=mul2(sign*probability,weight);
 resid_hi[row]=r.x;resid_lo[row]=r.y;
 // Normalize before reduction to reduce overflow in large datasets.
 loss[row]=(sw[row]/denom[0])*(max(u.x,0.0f)+log1p(exp(-abs(u.x))));
 // exp2x deliberately clamps beyond -80. Bound only that tail error;
 // other arithmetic and transcendental errors are not formally bounded.
 tail_error[row]=(abs(u.x)>80.0f)?(sw[row]/denom[0])*(feature_max*1.8048524e-35f):0.0f;
 }
}
""",
)

_logits = mx.fast.metal_kernel(
    name="mlxlearn_device_logistic_compensated_logits",
    header=_HEADER,
    input_names=["X", "center", "theta"],
    output_names=["hi", "lo"],
    source=r"""
uint t=thread_position_in_threadgroup.x,lane=t%32,sg=t/32;
uint row=threadgroup_position_in_grid.x*4+sg;
float2 dot=float2(0);
if(row<N){for(uint j=lane;j<D;j+=32){
 float2 dx=add2(float2(X[ulong(row)*D+j],0),float2(-center[j],0));
 dot=add2(dot,mul2(dx,float2(theta[j],0)));
}}
dot=sum2(dot);
if(lane==0 && row<N){dot=add2(dot,float2(theta[D],0));hi[row]=dot.x;lo[row]=dot.y;}
""",
)


def logits_parts(X, center, theta):
    """Evaluate centered logits with compensated GPU dot products."""
    n, d = X.shape
    return _logits(
        inputs=[X, center, theta], template=[("N", n), ("D", d)],
        grid=(((n + 3) // 4) * 128, 1, 1), threadgroup=(128, 1, 1),
        output_shapes=[(n,), (n,)], output_dtypes=[mx.float32, mx.float32],
    )


_backward = mx.fast.metal_kernel(
    name="mlxlearn_device_logistic_compensated_backward",
    header=_HEADER,
    input_names=["X", "theta", "resid_hi", "resid_lo", "denom", "l2"],
    output_names=["grad_hi", "grad_lo"],
    source=r"""
uint lane=thread_position_in_threadgroup.x;uint col=threadgroup_position_in_grid.x;
float2 g=float2(0);
for(uint row=lane;row<N;row+=32){
 float x=col<D?X[ulong(row)*D+col]:1.0f;
 g=add2(g,mul2(float2(x,0),float2(resid_hi[row],resid_lo[row])));
}
g=sum2(g);
if(lane==0){
 if(col<D)g=add2(g,mul2(float2(l2[0],l2[1]),float2(theta[col],0)));
 grad_hi[col]=g.x;grad_lo[col]=g.y;
}
""",
)



# Bound temporary high/low partial arrays and Metal grid dimensions. Larger
# inputs fall back to the original constant-workspace compensated kernel.
_MAX_COALESCED_WORKSPACE_BYTES = 64 * 1024 * 1024

_coalesced_partial = mx.fast.metal_kernel(
    name="mlxlearn_device_compensated_gradient_partial",
    header=_HEADER,
    input_names=["X", "resid_hi", "resid_lo"],
    output_names=["part_hi", "part_lo"],
    source=r"""
uint t=thread_position_in_threadgroup.x;
uint lane=t%32, sg=t/32;
uint col=threadgroup_position_in_grid.x*32+lane;
uint block=threadgroup_position_in_grid.y;
float2 value=float2(0.0f);
if(col<P){
 for(uint row=block*256+sg;row<min(uint(N),(block+1)*256);row+=4){
  float x=col<D?X[ulong(row)*D+col]:1.0f;
  value=add2(value,mul2(float2(x,0),float2(resid_hi[row],resid_lo[row])));
 }
}
threadgroup float2 scratch[128];
scratch[t]=value;
threadgroup_barrier(mem_flags::mem_threadgroup);
if(sg==0 && col<P){
 float2 out=add2(add2(scratch[lane],scratch[lane+32]),add2(scratch[lane+64],scratch[lane+96]));
 part_hi[ulong(block)*P+col]=out.x;
 part_lo[ulong(block)*P+col]=out.y;
}
""",
)

_coalesced_finish = mx.fast.metal_kernel(
    name="mlxlearn_device_compensated_gradient_finish",
    header=_HEADER,
    input_names=["part_hi", "part_lo", "theta", "l2"],
    output_names=["grad_hi", "grad_lo"],
    source=r"""
uint t=thread_position_in_threadgroup.x;
uint lane=t%32, sg=t/32;
uint col=threadgroup_position_in_grid.x*32+lane;
float2 value=float2(0.0f);
if(col<P){
 for(uint block=sg;block<B;block+=4){
  ulong i=ulong(block)*P+col;
  value=add2(value,float2(part_hi[i],part_lo[i]));
 }
}
threadgroup float2 scratch[128];
scratch[t]=value;
threadgroup_barrier(mem_flags::mem_threadgroup);
if(sg==0 && col<P){
 float2 g=add2(add2(scratch[lane],scratch[lane+32]),add2(scratch[lane+64],scratch[lane+96]));
 if(col<D)g=add2(g,mul2(float2(l2[0],l2[1]),float2(theta[col],0)));
 grad_hi[col]=g.x;
 grad_lo[col]=g.y;
}
""",
)


def _coalesced_backward(X, theta, resid_hi, resid_lo, denom, l2):
    """Return compensated original-coordinate gradient high/low parts.

    Residuals are already normalized; ``denom`` is unused.
    """
    del denom
    n, d = X.shape
    p = d + 1
    blocks = (n + 255) // 256
    groups = (p + 31) // 32
    hi, lo = _coalesced_partial(
        inputs=[X, resid_hi, resid_lo],
        template=[("N", n), ("D", d), ("P", p)],
        grid=(groups * 128, blocks, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(blocks, p), (blocks, p)],
        output_dtypes=[mx.float32, mx.float32],
    )
    return _coalesced_finish(
        inputs=[hi, lo, theta, l2],
        template=[("D", d), ("P", p), ("B", blocks)],
        grid=(groups * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(p,), (p,)],
        output_dtypes=[mx.float32, mx.float32],
    )



def _coalesced_workspace_supported(n: int, d: int) -> bool:
    blocks = (n + 255) // 256
    return 0 < blocks <= 65535 and blocks * (d + 1) * 8 <= _MAX_COALESCED_WORKSPACE_BYTES

def _split(value: float) -> tuple[float, float]:
    # float32 rounding without importing NumPy. struct is only used for two
    # scalar metadata values; no device arrays leave the GPU.
    import struct

    hi = struct.unpack("=f", struct.pack("=f", value))[0]
    lo = struct.unpack("=f", struct.pack("=f", value - hi))[0]
    return hi, lo


def score_original(X, center, theta, y, sw, sw_sum: float, l2: float):
    """Return objective and original-coordinate gradient as device arrays.

    The mathematical input is the original float32 design, not the rounded
    centered design. The intercept is in centered coordinates.
    """
    n, d = X.shape
    denom = mx.array(_split(sw_sum), dtype=mx.float32)
    penalty_scale = mx.array(_split(l2), dtype=mx.float32)
    hi, lo, losses, tail_errors = _forward(
        inputs=[X, center, theta, y, sw, denom],
        template=[("N", n), ("D", d)],
        grid=(((n + 3) // 4) * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(n,), (n,), (n,), (n,)],
        output_dtypes=[mx.float32, mx.float32, mx.float32, mx.float32],
    )
    if _coalesced_workspace_supported(n, d):
        gh, gl = _coalesced_backward(X, theta, hi, lo, denom, penalty_scale)
    else:
        gh, gl = _backward(
            inputs=[X, theta, hi, lo, denom, penalty_scale],
            template=[("N", n), ("D", d)],
            grid=((d + 1) * 32, 1, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(d + 1,), (d + 1,)],
            output_dtypes=[mx.float32, mx.float32],
        )
    objective = mx.sum(losses) + mx.array(0.5 * l2, mx.float32) * mx.sum(theta[:d] ** 2)
    gradient_norm = mx.max(mx.abs(gh + gl))
    # Factor 2 accounts conservatively for float32 rounding in this particular
    # tail sum. This is not a bound on all other numerical errors.
    tail_bound = 2.0 * mx.sum(tail_errors)
    return objective, gradient_norm, gh, gl, tail_bound
