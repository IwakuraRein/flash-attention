"""
cuDNN SDPA Shape and Dtype Validator
run `python shape_check.py --help` to see the available options

Exits with code 0 on success (no output), exits with code 1 and prints error on failure.
Time to run depends on CPU, but should be under 1 second.
"""

import argparse
import sys
import time
from typing import Tuple
import cudnn


parser = argparse.ArgumentParser(
    description="Validate SDPA shapes and dtypes for cuDNN",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--batch_size", "-b", type=int, default=1, help="Batch size")
parser.add_argument("--q_seqlen", "-sq", type=int, default=8192, help="Query sequence length")
parser.add_argument("--kv_seqlen", "-skv", type=int, default=8192, help="Key/Value sequence length")
parser.add_argument("--num_q_heads", "-hq", type=int, default=16, help="Number of query heads")
parser.add_argument("--num_kv_heads", "-hkv", type=int, default=8, help="Number of key/value heads")
parser.add_argument("--head_dim", "-d", type=int, default=128, help="Head dimension (for both QK and VO)")
parser.add_argument("--head_dim_qk", type=int, default=None, help="Head dimension for Q/K (overrides --head_dim)")
parser.add_argument("--head_dim_vo", type=int, default=None, help="Head dimension for V/O (overrides --head_dim)")
parser.add_argument(
    "--dtype", "-t", type=str, default="bfloat16",
    choices=["float16", "fp16", "bfloat16", "bf16", "float32", "fp32", "fp8", "fp8_e4m3", "fp8_e5m2"],
    help="Data type"
)
parser.add_argument(
    "--attn_mask", "-m", type=str, default="no_mask",
    choices=["no_mask", "top_left", "bottom_right"],
    help="Attention mask type"
)
parser.add_argument("--fwd_only", action="store_true", help="Only validate forward pass")
args = parser.parse_args()

# Preprocess arguments
head_dim_qk = args.head_dim_qk if args.head_dim_qk is not None else args.head_dim
head_dim_vo = args.head_dim_vo if args.head_dim_vo is not None else args.head_dim
dtype_map = {
    "float16": cudnn.data_type.HALF,
    "fp16": cudnn.data_type.HALF,
    "bfloat16": cudnn.data_type.BFLOAT16,
    "bf16": cudnn.data_type.BFLOAT16,
    "float32": cudnn.data_type.FLOAT,
    "fp32": cudnn.data_type.FLOAT,
    "fp8": cudnn.data_type.FP8_E4M3,
    "fp8_e4m3": cudnn.data_type.FP8_E4M3,
    "fp8_e5m2": cudnn.data_type.FP8_E5M2,
}
cudnn_dtype = dtype_map[args.dtype]


def compute_strides(dims: Tuple[int, ...]) -> Tuple[int, ...]:
    """Compute row-major (C-contiguous) strides for given dimensions."""
    strides = []
    stride = 1
    for dim in reversed(dims):
        strides.append(stride)
        stride *= dim
    return tuple(reversed(strides))


# def compute_strides_col_major(dims):
#     strides = []
#     stride = 1
#     for dim in dims:  # forward instead of reversed
#         strides.append(stride)
#         stride *= dim
#     return tuple(strides)


def make_tensor(graph: cudnn.pygraph, dims: Tuple[int, ...], data_type, name: str = ""):
    """Create a cuDNN tensor with given dimensions and data type."""
    strides = compute_strides(dims)
    return graph.tensor(dim=dims, stride=strides, data_type=data_type, name=name)


def validate_sdpa_shapes(
    batch_size: int,
    q_seqlen: int,
    kv_seqlen: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim_qk: int,
    head_dim_vo: int,
    dtype: cudnn.data_type,
    attn_mask: str = "no_mask",
    validate_backward: bool = True,
):
    
    is_fp8 = dtype in (cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2)
    attn_scale = head_dim_qk ** (-0.5)

    # [B, H, S, D]
    q_dims = (batch_size, num_q_heads, q_seqlen, head_dim_qk)
    k_dims = (batch_size, num_kv_heads, kv_seqlen, head_dim_qk)
    v_dims = (batch_size, num_kv_heads, kv_seqlen, head_dim_vo)
    o_dims = (batch_size, num_q_heads, q_seqlen, head_dim_vo)
    stats_dims = (batch_size, num_q_heads, q_seqlen, 1)
    scale_dims = (1, 1, 1, 1)

    # ========== Forward Graph Validation ==========
    try:
        graph_fwd = cudnn.pygraph(
            io_data_type=dtype,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
        )

        q_fwd = make_tensor(graph_fwd, q_dims, dtype, "Q")
        k_fwd = make_tensor(graph_fwd, k_dims, dtype, "K")
        v_fwd = make_tensor(graph_fwd, v_dims, dtype, "V")

        if is_fp8:
            descale_q = make_tensor(graph_fwd, scale_dims, cudnn.data_type.FLOAT, "descale_Q")
            descale_k = make_tensor(graph_fwd, scale_dims, cudnn.data_type.FLOAT, "descale_K")
            descale_v = make_tensor(graph_fwd, scale_dims, cudnn.data_type.FLOAT, "descale_V")
            descale_s = make_tensor(graph_fwd, scale_dims, cudnn.data_type.FLOAT, "descale_S")
            scale_s = make_tensor(graph_fwd, scale_dims, cudnn.data_type.FLOAT, "scale_S")
            scale_o = make_tensor(graph_fwd, scale_dims, cudnn.data_type.FLOAT, "scale_O")

            o_fwd, stats_fwd, amax_s_fwd, amax_o_fwd = graph_fwd.sdpa_fp8(
                q=q_fwd, k=k_fwd, v=v_fwd,
                descale_q=descale_q, descale_k=descale_k, descale_v=descale_v,
                descale_s=descale_s, scale_s=scale_s, scale_o=scale_o,
                is_inference=False,
                attn_scale=attn_scale,
                diagonal_alignment=(
                    cudnn.diagonal_alignment.BOTTOM_RIGHT
                    if attn_mask == "bottom_right"
                    else cudnn.diagonal_alignment.TOP_LEFT
                ),
                right_bound=None if attn_mask == "no_mask" else 0,
            )

            o_fwd.set_output(True).set_dim(o_dims).set_stride(compute_strides(o_dims))
            stats_fwd.set_output(True).set_dim(stats_dims).set_stride(compute_strides(stats_dims))
            amax_s_fwd.set_output(True).set_dim(scale_dims).set_stride(compute_strides(scale_dims))
            amax_o_fwd.set_output(True).set_dim(scale_dims).set_stride(compute_strides(scale_dims))
        else:
            o_fwd, stats_fwd = graph_fwd.sdpa(
                q=q_fwd, k=k_fwd, v=v_fwd,
                is_inference=False,
                attn_scale=attn_scale,
                diagonal_alignment=(
                    cudnn.diagonal_alignment.BOTTOM_RIGHT
                    if attn_mask == "bottom_right"
                    else cudnn.diagonal_alignment.TOP_LEFT
                ),
                diagonal_band_right_bound=None if attn_mask == "no_mask" else 0,
            )

            o_fwd.set_output(True).set_dim(o_dims).set_stride(compute_strides(o_dims))
            stats_fwd.set_output(True).set_dim(stats_dims).set_stride(compute_strides(stats_dims))

        graph_fwd.validate()

    except Exception as e:
        print(f"[ERROR] Forward graph validation failed: {e}", file=sys.stderr)
        sys.exit(1)

    # ========== Backward Graph Validation ==========
    if validate_backward:
        try:
            graph_bwd = cudnn.pygraph(
                io_data_type=dtype,
                intermediate_data_type=cudnn.data_type.FLOAT,
                compute_data_type=cudnn.data_type.FLOAT,
            )

            q_bwd = make_tensor(graph_bwd, q_dims, dtype, "Q")
            k_bwd = make_tensor(graph_bwd, k_dims, dtype, "K")
            v_bwd = make_tensor(graph_bwd, v_dims, dtype, "V")
            o_bwd = make_tensor(graph_bwd, o_dims, dtype, "O")
            dO_bwd = make_tensor(graph_bwd, o_dims, dtype, "dO")
            stats_bwd = make_tensor(graph_bwd, stats_dims, cudnn.data_type.FLOAT, "stats")

            if is_fp8:
                descale_q = make_tensor(graph_bwd, scale_dims, cudnn.data_type.FLOAT, "descale_Q")
                descale_k = make_tensor(graph_bwd, scale_dims, cudnn.data_type.FLOAT, "descale_K")
                descale_v = make_tensor(graph_bwd, scale_dims, cudnn.data_type.FLOAT, "descale_V")
                descale_o = make_tensor(graph_bwd, scale_dims, cudnn.data_type.FLOAT, "descale_O")
                descale_dO = make_tensor(graph_bwd, scale_dims, cudnn.data_type.FLOAT, "descale_dO")
                descale_s = make_tensor(graph_bwd, scale_dims, cudnn.data_type.FLOAT, "descale_S")
                descale_dP = make_tensor(graph_bwd, scale_dims, cudnn.data_type.FLOAT, "descale_dP")
                scale_s = make_tensor(graph_bwd, scale_dims, cudnn.data_type.FLOAT, "scale_S")
                scale_dQ = make_tensor(graph_bwd, scale_dims, cudnn.data_type.FLOAT, "scale_dQ")
                scale_dK = make_tensor(graph_bwd, scale_dims, cudnn.data_type.FLOAT, "scale_dK")
                scale_dV = make_tensor(graph_bwd, scale_dims, cudnn.data_type.FLOAT, "scale_dV")
                scale_dP = make_tensor(graph_bwd, scale_dims, cudnn.data_type.FLOAT, "scale_dP")

                (
                    dQ_bwd, dK_bwd, dV_bwd,
                    amax_dQ, amax_dK, amax_dV, amax_dP,
                ) = graph_bwd.sdpa_fp8_backward(
                    q=q_bwd, k=k_bwd, v=v_bwd, o=o_bwd, dO=dO_bwd, stats=stats_bwd,
                    descale_q=descale_q, descale_k=descale_k, descale_v=descale_v,
                    descale_o=descale_o, descale_dO=descale_dO, descale_s=descale_s,
                    descale_dP=descale_dP, scale_s=scale_s,
                    scale_dQ=scale_dQ, scale_dK=scale_dK, scale_dV=scale_dV, scale_dP=scale_dP,
                    attn_scale=attn_scale,
                    use_causal_mask=attn_mask != "no_mask" and attn_mask != "bottom_right",
                    use_causal_mask_bottom_right=attn_mask == "bottom_right",
                )

                dQ_bwd.set_output(True).set_dim(q_dims).set_stride(compute_strides(q_dims))
                dK_bwd.set_output(True).set_dim(k_dims).set_stride(compute_strides(k_dims))
                dV_bwd.set_output(True).set_dim(v_dims).set_stride(compute_strides(v_dims))
                amax_dQ.set_output(True).set_dim(scale_dims).set_stride(compute_strides(scale_dims))
                amax_dK.set_output(True).set_dim(scale_dims).set_stride(compute_strides(scale_dims))
                amax_dV.set_output(True).set_dim(scale_dims).set_stride(compute_strides(scale_dims))
                amax_dP.set_output(True).set_dim(scale_dims).set_stride(compute_strides(scale_dims))
            else:
                dQ_bwd, dK_bwd, dV_bwd = graph_bwd.sdpa_backward(
                    q=q_bwd, k=k_bwd, v=v_bwd, o=o_bwd, dO=dO_bwd, stats=stats_bwd,
                    attn_scale=attn_scale,
                    diagonal_alignment=(
                        cudnn.diagonal_alignment.BOTTOM_RIGHT
                        if attn_mask == "bottom_right"
                        else cudnn.diagonal_alignment.TOP_LEFT
                    ),
                    diagonal_band_right_bound=None if attn_mask == "no_mask" else 0,
                )

                dQ_bwd.set_output(True).set_dim(q_dims).set_stride(compute_strides(q_dims))
                dK_bwd.set_output(True).set_dim(k_dims).set_stride(compute_strides(k_dims))
                dV_bwd.set_output(True).set_dim(v_dims).set_stride(compute_strides(v_dims))

            graph_bwd.validate()

        except Exception as e:
            print(f"[ERROR] Backward graph validation failed: {e}", file=sys.stderr)
            sys.exit(1)

tic = time.perf_counter()
validate_sdpa_shapes(
    batch_size=args.batch_size,
    q_seqlen=args.q_seqlen,
    kv_seqlen=args.kv_seqlen,
    num_q_heads=args.num_q_heads,
    num_kv_heads=args.num_kv_heads,
    head_dim_qk=head_dim_qk,
    head_dim_vo=head_dim_vo,
    dtype=cudnn_dtype,
    attn_mask=args.attn_mask,
    validate_backward=not args.fwd_only,
)
toc = time.perf_counter()
print(f"[INFO] Shape validation took {toc - tic:.2f} seconds")