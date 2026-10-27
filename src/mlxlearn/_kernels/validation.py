"""Copy a borrowed float32 design into owned GPU storage while validating it."""

import mlx.core as mx

_copy_finite = mx.fast.metal_kernel(
    name="mlxlearn_copy_finite",
    input_names=["X"],
    output_names=["owned", "valid"],
    source=r"""
uint t=thread_position_in_threadgroup.x;
ulong block=ulong(threadgroup_position_in_grid.x)+
    ulong(threadgroup_position_in_grid.y)*BLOCK_COLUMNS;
ulong size=ulong(N)*D;
bool finite=true;
for(uint k=0;k<4;++k){
    ulong i=block*1024+t+k*256;
    if(i<size){float value=X[i];owned[i]=value;finite=finite && isfinite(value);}
}
bool lane_valid=simd_all(finite);
threadgroup uint groups[8];
if((t%32)==0)groups[t/32]=uint(lane_valid);
threadgroup_barrier(mem_flags::mem_threadgroup);
if(t==0 && block<BLOCKS){uint result=1;for(uint j=0;j<8;++j)result &= groups[j];valid[block]=result;}
""",
)


def copy_and_check(X):
    """Return an independent, evaluated copy and a host finiteness flag.

    The borrowed input remains alive only while the copy is made. Evaluating
    both outputs before returning establishes an ownership boundary, even when
    a caller mutates its NumPy array immediately afterwards.
    """
    blocks = (X.size + 1023) // 1024
    block_columns = min(blocks, 65535)
    owned, valid = _copy_finite(
        inputs=[X],
        template=[("N", X.shape[0]), ("D", X.shape[1]),
                  ("BLOCKS", blocks), ("BLOCK_COLUMNS", block_columns)],
        grid=(block_columns * 256, (blocks + block_columns - 1) // block_columns, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[X.shape, (blocks,)],
        output_dtypes=[mx.float32, mx.uint32],
    )
    finite = mx.all(valid != 0)
    mx.eval(owned, finite)
    return owned, bool(finite)
