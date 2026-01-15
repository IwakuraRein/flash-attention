import argparse

from collections import namedtuple
from functools import partial
from itertools import product
import math
import sys
from typing import NamedTuple
import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import subprocess

try:
    import cudnn
except ImportError:
    cudnn = None
# cudnn = None

Timing = NamedTuple('timing', [('mean', float)])


from einops import rearrange, repeat

from flash_attn.cute.interface import flash_attn_func as flash_attn_func_python
from flash_attn.cute.interface import flash_attn_varlen_func as flash_attn_varlen_func_python

assert torch.cuda.get_device_capability()[0] >= 10, "This benchmark only supports sm100."

flash_attn_func = None

import cuda.bindings.driver as cuda_driver
import cuda.bindings.runtime as cuda_runtime
from triton.testing import do_bench

import cutlass.cute.testing as testing

torch_stream = torch.cuda.Stream()
driver_stream = cuda_driver.CUstream(torch_stream.cuda_stream)

var_seq_drop_token_num = [0] * 128 # TODO: replace with values close to production
def unpad(x: torch.Tensor, batch_size: int) -> torch.Tensor:
    sequences = []
    for i in range(batch_size):
        sequences.append(x[i, :-var_seq_drop_token_num[i], :, :])
    return torch.cat(sequences, dim=0)

var_seq_drop_token_num_cum = torch.tensor([0] * 128, device='cuda', dtype=torch.int32)

def generate_kernel_arguments(
        batch_size,
        seqlen_q,
        seqlen,
        nheads,
        nheads_kv,
        headdim,
        headdim_v,
        V_colmajor,
        dtype,
        dtype_gen,
        has_backward,
        causal,
        window_size,
        varlen,
        deterministic,
        device,
        is_cudnn = False,
    ):
    q = torch.randn(batch_size, seqlen_q, nheads, headdim, device=device, dtype=dtype_gen, requires_grad=has_backward)
    k = torch.randn(batch_size, seqlen, nheads_kv, headdim, device=device, dtype=dtype_gen, requires_grad=has_backward)
    v = torch.randn(batch_size, seqlen, nheads_kv, headdim_v, device=device, dtype=dtype_gen, requires_grad=has_backward)
    q, k, v = [x.detach().to(dtype).requires_grad_(has_backward) for x in [q, k, v]]
    v_colmajor = v.detach().transpose(-1, -3).contiguous().transpose(-1, -3).requires_grad_(has_backward)
    v_fa3 = v if not V_colmajor else v_colmajor

    if is_cudnn or varlen:
        q_unpad, k_unpad, v_unpad = [rearrange(x.detach(), "b s h d -> (b s) h d").requires_grad_(has_backward) for x in [q, k, v]]
        cu_seqlens_q = torch.arange(batch_size + 1, device=device, dtype=torch.int32) * seqlen_q
        cu_seqlens_k = torch.arange(batch_size + 1, device=device, dtype=torch.int32) * seqlen
        # q_unpad, k_unpad, v_unpad = [unpad(x, batch_size).requires_grad_(has_backward) for x in [q, k, v]]
        # cu_seqlens_q = torch.arange(batch_size + 1, device=device, dtype=torch.int32) * seqlen_q - var_seq_drop_token_num_cum[:batch_size+1]
        # cu_seqlens_k = torch.arange(batch_size + 1, device=device, dtype=torch.int32) * seqlen - var_seq_drop_token_num_cum[:batch_size+1]

        return testing.JitArguments(
            q if is_cudnn or not varlen else q_unpad,
            k if is_cudnn or not varlen else k_unpad,
            v_fa3 if is_cudnn or not varlen else v_unpad,
            causal=causal,
            window_size=window_size,
            cu_seqlens_q=cu_seqlens_q if varlen else None,
            cu_seqlens_k=cu_seqlens_k if varlen else None,
            max_seqlen_q=seqlen_q if varlen else None, # FA4
            max_seqlen_k=seqlen if varlen else None, # FA4
            deterministic=deterministic
        )
    else:
        return testing.JitArguments(
            q,
            k,
            v_fa3,
            causal=causal,
            window_size=window_size,
        )


def get_iterations(batch_size, seqlen_q, nheads):
    gemm_bs = batch_size * nheads
    if gemm_bs >= 128 and seqlen_q >= 32768:
        return 2, 3
    if gemm_bs >= 128 and seqlen_q >= 16384:
        return 3, 5
    return 5, 10


def time_cute_fwd(func, workspace_generator, warmup_iterations=5, iterations=10):
    args = workspace_generator.args
    batch_size = args[0]
    seqlen_q = args[1]
    nheads = args[3]
    nheads_kv = args[4]
    headdim = args[5]
    one_workspace_bytes = batch_size * seqlen_q * (nheads + 2 * nheads_kv) * headdim * 2

    workspace_count = testing.get_workspace_count(one_workspace_bytes, 5, 10)

    benchmark_fn = partial(testing.benchmark,
                           func,
                           warmup_iterations=warmup_iterations,
                           iterations=iterations,
                           workspace_generator=workspace_generator,
                           workspace_count=workspace_count,
                           )

    try:
        with torch.cuda.stream(torch_stream):
            avg_time_us = benchmark_fn(
                stream=driver_stream,
                use_cuda_graphs=True,
            )
    except torch.AcceleratorError:
        print("CUDA graph capture failed, falling back to non-graph mode.", workspace_generator.args, file=sys.stderr)
        cuda_runtime.cudaStreamEndCapture(driver_stream)
        avg_time_us = benchmark_fn()

    return Timing(avg_time_us * 1e-6)


def time_cute_bwd(func, workspace_generator, warmup_iterations=5, iterations=10):
    args = workspace_generator.args
    batch_size = args[0]
    seqlen_q = args[1]
    nheads = args[3]
    nheads_kv = args[4]
    headdim = args[5]
    one_workspace_bytes = batch_size * seqlen_q * (2 * nheads + 2 * nheads_kv) * headdim * 2 * 2

    workspace_count = testing.get_workspace_count(one_workspace_bytes, 5, 10)

    with torch.cuda.stream(torch_stream):
        bwd_workspaces = []
        for _ in range(workspace_count):
            fwd_workspace = workspace_generator()
            out, lse = func(*fwd_workspace.args, **fwd_workspace.kwargs)
            grad = torch.randn_like(out)
            bwd_workspaces.append(testing.JitArguments(out, grad, fwd_workspace))

        def func_bwd(out, grad, fwd_workspace):
            # Set .grad to None to avoid extra operation of gradient accumulation
            for x in fwd_workspace.args + tuple(fwd_workspace.kwargs.values()):
                if isinstance(x, torch.Tensor):
                    x.grad = None
            out.backward(grad, retain_graph=True)

        avg_time_us = testing.benchmark(
            func_bwd,
            warmup_iterations=warmup_iterations,
            iterations=iterations,
            stream=driver_stream,
            workspace_generator=partial(next, iter(bwd_workspaces)),
            workspace_count=workspace_count,
            use_cuda_graphs=True,
        )
    return Timing(avg_time_us * 1e-6)


def time_fwd(func, *args, repeats=30, verbose=True, desc="", **kwargs):
    # # Warmup
    # for _ in range(5):
    #     func(*args, **kwargs)
    # time.sleep(1)
    # return benchmark_forward(func, *args, **kwargs, repeats=repeats, verbose=verbose, desc=desc)[1]
    # s = torch.cuda.Stream()
    # s.wait_stream(torch.cuda.current_stream())
    # with torch.cuda.stream(s):
    #     for _ in range(2):
    #         out = func(*args, **kwargs)
    # torch.cuda.current_stream().wait_stream(s)
    # graph = torch.cuda.CUDAGraph()
    # with torch.cuda.graph(graph):
    #     out = func(*args, **kwargs)
    # time_f = benchmark_forward(lambda: graph.replay(), repeats=repeats, verbose=verbose, desc=desc)
    # # return time_f[1].mean
    # return time_f[1]
    return Timing(do_bench(lambda: func(*args, **kwargs), warmup=5, rep=repeats) * 1e-3)


def flops(batch, nheads, seqlen_q, seqlen_k, headdim, headdim_v, causal=False, window_size=(None, None)):
    if causal:
        avg_seqlen = (max(0, seqlen_k - seqlen_q) + seqlen_k) / 2
    else:
        if window_size == (None, None):
            avg_seqlen = seqlen_k
        else:
            row_idx = torch.arange(seqlen_q, device='cuda')
            col_left = (
                torch.maximum(row_idx + seqlen_k - seqlen_q - window_size[0], torch.tensor(0)) if window_size[0] is not None
                else torch.zeros_like(row_idx)
            )
            col_right = (
                torch.minimum(row_idx + seqlen_k - seqlen_q + window_size[1], torch.tensor(seqlen_k - 1)) if window_size[1] is not None
                else torch.full_like(row_idx, seqlen_k - 1)
            )
            avg_seqlen = (col_right - col_left + 1).float().mean().item()
    return batch * nheads * 2 * seqlen_q * avg_seqlen * (headdim + headdim_v)


def mem_size_fwd(batch, nheads, nheads_kv, seqlen_q, seqlen_k, head_dim, head_dim_v):
    elem_size = 2
    total_size = 0
    total_size += batch * seqlen_q * nheads * head_dim * elem_size # Q
    total_size += batch * seqlen_k * nheads_kv * head_dim * elem_size # K
    total_size += batch * seqlen_k * nheads_kv * head_dim_v * elem_size # V
    total_size += batch * seqlen_q * nheads * head_dim_v * elem_size # O
    return total_size


def mem_size_bwd(batch, nheads, nheads_kv, seqlen_q, seqlen_k, head_dim, head_dim_v):
    elem_size = 2
    accum_elem_size = 4

    total_size = 0

    # preprocess
    total_size += batch * seqlen_q * nheads * head_dim_v * elem_size * 2 # O, dO
    total_size += batch * seqlen_q * nheads * accum_elem_size * 3 # dPsum, LSE, LSE_log2
    total_size += batch * seqlen_q * nheads * head_dim * accum_elem_size # dQ_accum
    if nheads_kv != nheads:
        total_size += batch * seqlen_k * nheads_kv * head_dim * accum_elem_size # dK_accum
        total_size += batch * seqlen_k * nheads_kv * head_dim_v * accum_elem_size # dV_accum

    # main
    total_size += batch * seqlen_q * nheads * head_dim * elem_size # Q
    total_size += batch * seqlen_k * nheads_kv * head_dim * elem_size # K
    total_size += batch * seqlen_k * nheads_kv * head_dim_v * elem_size # V
    total_size += batch * seqlen_q * nheads * head_dim_v * elem_size # dO
    total_size += batch * seqlen_q * nheads * accum_elem_size * 2 # dPsum, LSE_log2
    total_size += batch * seqlen_q * nheads * head_dim * accum_elem_size # dQ_accum
    if nheads_kv != nheads:
        total_size += batch * seqlen_k * nheads_kv * head_dim * accum_elem_size # dK_accum
        total_size += batch * seqlen_k * nheads_kv * head_dim_v * accum_elem_size # dV_accum
    else:
        total_size += batch * seqlen_k * nheads_kv * head_dim * elem_size # dK
        total_size += batch * seqlen_k * nheads_kv * head_dim_v * elem_size # dV

    # postprocess
    total_size += batch * seqlen_q * nheads * head_dim * (elem_size + accum_elem_size) # dQ, dQ_accum
    if nheads_kv != nheads:
        total_size += batch * seqlen_k * nheads_kv * head_dim * (elem_size + accum_elem_size) # dK, dK_accum
        total_size += batch * seqlen_k * nheads_kv * head_dim_v * (elem_size + accum_elem_size) # dV, dV_accum

    return total_size


def mem_throughput(time, **tensors):
    total_size = sum(float(x.size) for x in tensors.values()) * 1e-12 # TB
    return total_size / time # TB/s


def convert_to_cudnn_type(torch_type):
    if torch_type == torch.float16:
        return cudnn.data_type.HALF
    elif torch_type == torch.bfloat16:
        return cudnn.data_type.BFLOAT16
    elif torch_type == torch.float32:
        return cudnn.data_type.FLOAT
    elif torch_type == torch.int32:
        return cudnn.data_type.INT32
    elif torch_type == torch.int64:
        return cudnn.data_type.INT64
    else:
        raise ValueError("Unsupported tensor data type.")


def cudnn_spda_setup(q, k, v, causal=False, window_size_left=None, varlen=False):
    device = q.device
    b, nheads, seqlen_q, headdim = q.shape
    _, nheads_k, seqlen_k, _ = k.shape
    headdim_v = v.shape[-1]
    assert v.shape == (b, nheads_k, seqlen_k, headdim_v)
    assert cudnn is not None, 'CUDNN is not available'
    q_gpu, k_gpu, v_gpu = q, k, v
    o_gpu = torch.empty((b, nheads, seqlen_q, headdim_v), dtype=q.dtype, device=device)
    stats_gpu = torch.empty(b, nheads, seqlen_q, 1, dtype=torch.float32, device=device)
    graph = cudnn.pygraph(
        io_data_type=convert_to_cudnn_type(q.dtype),
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    q = graph.tensor_like(q_gpu.detach())
    k = graph.tensor_like(k_gpu.detach())
    v = graph.tensor_like(v_gpu.detach())

    if varlen:
        drop_num = torch.tensor(var_seq_drop_token_num[:b], device=device, dtype=torch.int32)
        seq_len_q_gpu = (torch.ones(b, device=device, dtype=torch.int32) * seqlen_q - drop_num).reshape(b, 1, 1, 1)
        seq_len_kv_gpu = (torch.ones(b, device=device, dtype=torch.int32) * seqlen_k - drop_num).reshape(b, 1, 1, 1)
        seq_len_q = graph.tensor_like(seq_len_q_gpu)
        seq_len_kv = graph.tensor_like(seq_len_kv_gpu)

    o, stats = graph.sdpa(
        name="sdpa",
        q=q,
        k=k,
        v=v,
        use_padding_mask=varlen,
        seq_len_q=seq_len_q if varlen else None,
        seq_len_kv=seq_len_kv if varlen else None,
        is_inference=False,
        attn_scale=1.0 / math.sqrt(headdim),
        # use_causal_mask_bottom_right=causal or window_size_left is not None,
        use_causal_mask=causal or window_size_left is not None,
        sliding_window_length=window_size_left if window_size_left is not None and not causal else None,
    )

    o.set_output(True).set_dim(o_gpu.shape).set_stride(o_gpu.stride())
    stats.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()

    variant_pack = {
        q: q_gpu,
        k: k_gpu,
        v: v_gpu,
        o: o_gpu,
        stats: stats_gpu,
    }
    if varlen:
        variant_pack[seq_len_q] = seq_len_q_gpu
        variant_pack[seq_len_kv] = seq_len_kv_gpu

    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    def run(*args, **kwargs):
        graph.execute(variant_pack, workspace)
        return o_gpu

    return run


def cudnn_spda_bwd_setup(
        q,
        k,
        v,
        o,
        g,
        lse,
        causal=False,
        window_size_left=None,
        varlen=False,
        deterministic=False,
    ):
    device = q.device
    b, nheads, seqlen_q, headdim = q.shape
    _, nheads_k, seqlen_k, _ = k.shape
    headdim_v = v.shape[-1]
    assert v.shape == (b, nheads_k, seqlen_k, headdim_v)
    assert g.shape == (b, nheads, seqlen_q, headdim_v)
    assert o.shape == (b, nheads, seqlen_q, headdim_v)
    assert lse.shape == (b, nheads, seqlen_q, 1)
    assert cudnn is not None, 'CUDNN is not available'
    q_gpu, k_gpu, v_gpu, o_gpu, g_gpu = q, k, v, o, g
    dq_gpu = torch.empty_like(q_gpu)
    dk_gpu = torch.empty_like(k_gpu)
    dv_gpu = torch.empty_like(v_gpu)
    graph = cudnn.pygraph(
        io_data_type=convert_to_cudnn_type(q.dtype),
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    q = graph.tensor_like(q_gpu.detach())
    k = graph.tensor_like(k_gpu.detach())
    v = graph.tensor_like(v_gpu.detach())
    o = graph.tensor_like(o_gpu.detach())
    g = graph.tensor_like(g_gpu.detach())
    stats = graph.tensor_like(lse.detach())
    if varlen:
        drop_num = torch.tensor(var_seq_drop_token_num[:b], device=device, dtype=torch.int32)
        seq_len_q_gpu = (torch.ones(b, device=device, dtype=torch.int32) * seqlen_q - drop_num).reshape(b, 1, 1, 1)
        seq_len_kv_gpu = (torch.ones(b, device=device, dtype=torch.int32) * seqlen_k - drop_num).reshape(b, 1, 1, 1)
        seq_len_q = graph.tensor_like(seq_len_q_gpu.detach())
        seq_len_kv = graph.tensor_like(seq_len_kv_gpu.detach())

    dq, dk, dv = graph.sdpa_backward(
        name="sdpa_backward",
        q=q,
        k=k,
        v=v,
        o=o,
        dO=g,
        stats=stats,
        use_padding_mask=varlen,
        seq_len_q=seq_len_q if varlen else None,
        seq_len_kv=seq_len_kv if varlen else None,
        attn_scale=1.0 / math.sqrt(headdim),
        # use_causal_mask_bottom_right=causal or window_size_left is not None,
        use_causal_mask=causal or window_size_left is not None,
        sliding_window_length=window_size_left if window_size_left is not None and not causal else None,
        use_deterministic_algorithm=deterministic,
    )

    dq.set_output(True).set_dim(dq_gpu.shape).set_stride(dq_gpu.stride())
    dk.set_output(True).set_dim(dk_gpu.shape).set_stride(dk_gpu.stride())
    dv.set_output(True).set_dim(dv_gpu.shape).set_stride(dv_gpu.stride())

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()

    variant_pack = {
        q: q_gpu,
        k: k_gpu,
        v: v_gpu,
        o: o_gpu,
        g: g_gpu,
        stats: lse,
        dq: dq_gpu,
        dk: dk_gpu,
        dv: dv_gpu,
    }
    if varlen:
        variant_pack[seq_len_q] = seq_len_q_gpu
        variant_pack[seq_len_kv] = seq_len_kv_gpu


    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    def run(*args, **kwargs):
        graph.execute(variant_pack, workspace)
        return dq_gpu, dk_gpu, dv_gpu

    return run


torch.manual_seed(0)

def benchmark_attn(
    run_fa4 = True,
    run_cudnn = False,
    dropout_p = 0.0,
    causal = False,
    window_size = (None, None),
    dtype = torch.bfloat16,
    verbose = False,
    varlen = False,
    has_backward = True,
    page_size = None,
    softcap = 0.0,
    V_colmajor = False,
    deterministic = False,
    batch_size = 2,
    seqlen = 8192,
    nheads = 16,
    nheads_kv = 16,
    headdim = 128,
    output_file = None
):
    time_f = {}
    time_b = {}
    device = 'cuda'
    dtype_gen = torch.bfloat16 if dtype == torch.float8_e4m3fn else dtype
    headdim_v = 128 if headdim == 192 else headdim
    # headdim_v = 512
    has_qv = headdim == 64 and headdim_v == 512
    # has_qv = False
    # sinks = torch.randn(nheads, dtype=torch.bfloat16, device=device)
    sinks = None

    if verbose:
        print(f'  {batch_size=}, {seqlen=}, {nheads=}, {headdim=}, {headdim_v=}, {has_qv=}, {has_backward=}')
        print(f'  {dtype=}, {dtype_gen=}, {device=}')
        print(f'  {V_colmajor=}, {deterministic=}, {softcap=}')
        print(f'  {page_size=}, {varlen=}, {has_backward=}')
        print(f'  {dropout_p=}, {causal=}')
        print(f'  {verbose=}')
    else:
        print(f"  {batch_size=}, {nheads=}, {nheads_kv=}, {seqlen=}, {headdim=}, {window_size=}, {causal=}, {varlen=}, {deterministic=}")

    output_config = f'{batch_size},{nheads},{nheads_kv},{seqlen},{headdim},{window_size[0]},{causal},{varlen},{deterministic}'

    # for batch_size, seqlen in bs_seqlen_vals:
    if True:
        seqlen_q = seqlen

        fwd_flop = flops(batch_size, nheads, seqlen_q, seqlen, headdim if not has_qv else headdim + headdim_v, headdim_v, causal=causal, window_size=window_size)
        fwd_mem_size = mem_size_fwd(batch_size, nheads, nheads_kv, seqlen_q, seqlen, headdim, headdim_v)
        has_backward = has_backward
        if has_backward:
            bwd_mem_size = mem_size_bwd(batch_size, nheads, nheads_kv, seqlen_q, seqlen, headdim, headdim_v)

        def log_result(kernel, fwd_time, bwd_time):
            fwd_flops = fwd_flop / fwd_time.mean * 1e-12
            fwd_tbps = fwd_mem_size / fwd_time.mean * 1e-12
            if has_backward and bwd_time.mean > 0:
                bwd_flops = 2.5 * fwd_flop / bwd_time.mean * 1e-12
                bwd_tbps = bwd_mem_size / bwd_time.mean * 1e-12

            print(f'\t{kernel}:')
            print(f'\t\tForward: {fwd_time.mean * 1e3:.3f} ms, {fwd_flops:.1f} TFLOPS, {fwd_tbps:.3f} TB/s')
            if has_backward and bwd_time.mean > 0:
                print(f'\t\tBackward: {bwd_time.mean * 1e3:.3f} ms, {bwd_flops:.1f} TFLOPS, {bwd_tbps:.3f} TB/s')

            if output_file is not None:
                output_file.write(f'{output_config},{kernel},{fwd_time.mean * 1e3:.3f},{fwd_flops:.1f},{fwd_tbps:.3f}')
                if has_backward and bwd_time.mean > 0:
                    output_file.write(f',{bwd_time.mean * 1e3:.3f},{bwd_flops:.1f},{bwd_tbps:.3f}')
                else:
                    output_file.write(',,,')
                output_file.write('\n')
                output_file.flush()

        g = torch.randn(batch_size, seqlen_q, nheads, headdim_v, device=device, dtype=dtype_gen)
        o = torch.randn(batch_size, seqlen_q, nheads, headdim_v, device=device, dtype=dtype_gen)
        stats = torch.randn(batch_size, seqlen_q, nheads, 1, device=device, dtype=torch.float32)

        warmup_repeats, repeats = get_iterations(batch_size, seqlen_q, nheads)

        run_fa4 = run_fa4 and flash_attn_func_python is not None
        run_cudnn = run_cudnn and cudnn is not None and headdim <= 256 and dtype != torch.float8_e4m3fn

        if run_cudnn:
            q, k, v = generate_kernel_arguments(
                batch_size,
                seqlen_q,
                seqlen,
                nheads,
                nheads_kv,
                headdim,
                headdim_v,
                V_colmajor,
                dtype,
                dtype_gen,
                has_backward,
                causal,
                window_size,
                varlen,
                deterministic,
                device,
                True,
            ).args
            cudnn_spda = cudnn_spda_setup(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=causal, window_size_left=window_size[0], varlen=varlen)
            if has_backward and headdim == headdim_v:
                cudnn_spda_bwd = cudnn_spda_bwd_setup(
                    q.transpose(1, 2),
                    k.transpose(1, 2),
                    v.transpose(1, 2),
                    o.transpose(1, 2),
                    g.transpose(1, 2),
                    stats.transpose(1, 2),
                    causal=causal,
                    window_size_left=window_size[0],
                    varlen=varlen,
                    deterministic=deterministic,
                )
            time.sleep(1) # Sleep to avoid residual power throttling from the previous benchmark
            m2 = time_fwd(cudnn_spda, warmup_repeats=warmup_repeats, repeats=repeats, verbose=verbose, desc='CuDNN')
            time_f[(causal, headdim, batch_size, seqlen), "cuDNN"] = m2.mean
            m2b = Timing(0.0)
            if has_backward:
                time.sleep(1)
                m2b = time_fwd(cudnn_spda_bwd, warmup_repeats=warmup_repeats, repeats=repeats, verbose=verbose, desc='CuDNN')
                time_b[(causal, headdim, batch_size, seqlen), "cuDNN"] = m2b.mean
            log_result("cuDNN", m2, m2b)

        if run_fa4:
            workspace_generator = partial(
                generate_kernel_arguments,
                batch_size,
                seqlen_q,
                seqlen,
                nheads,
                nheads_kv,
                headdim,
                headdim_v,
                V_colmajor,
                dtype,
                dtype_gen,
                has_backward,
                causal,
                window_size,
                varlen,
                deterministic,
                device,
                False
            )
            fn = flash_attn_func_python if not varlen else flash_attn_varlen_func_python
            m1_py = time_cute_fwd(fn, workspace_generator, warmup_repeats, repeats)
            m1b_py = Timing(0.0)
            if has_backward:
                m1b_py = time_cute_bwd(fn, workspace_generator, warmup_repeats, repeats)
            log_result("FA4", m1_py, m1b_py)


def sweep(output_file_name, run_fa4, run_cudnn):
    with open(output_file_name, 'w') as output_file:
        output_file.write(f'bs,nheads,nheads_kv,seqlen,headdim,window_size,causal,varlen,deterministic,kernel,fwd_time,fwd_flops,fwd_TBps,bwd_time,bwd_flops,bwd_TBps\n')
        for batch_size, nheads, seqlen, headdim, window_size, causal, gqa, varlen, deterministic in product(
            [1],
            [128],
            [512, 1024, 4096, 8192, 16384, 32768],
            [128],
            [(None, None)],
            [True],
            [8],
            [True],
            [False, True],
        ):
            nheads_kv = nheads // gqa
            benchmark_attn(
                run_fa4 = run_fa4,
                run_cudnn = run_cudnn,
                batch_size = batch_size,
                headdim = headdim,
                nheads = nheads,
                nheads_kv = nheads_kv,
                seqlen = seqlen,
                causal = causal,
                window_size = window_size,
                varlen = varlen,
                deterministic=deterministic,
                output_file = output_file,
            )
            if run_cudnn:
                # warmup_repeats, repeats = get_iterations(batch_size, seqlen, nheads)
                warmup_repeats, repeats = 5, 10
                cmd = [
                    sys.executable,  # Use the same Python interpreter
                    "benchmark_single_sdpa.py",
                    "--batch_size", str(batch_size),
                    "--head_dim", str(headdim),
                    "--num_q_heads", str(nheads),
                    "--num_kv_heads", str(nheads_kv),
                    "--q_seqlen", str(seqlen),
                    "--kv_seqlen", str(seqlen),
                    "--attn_mask", "top_left" if causal else "no_mask",
                    "--data_type", "bfloat16",
                    "--sdpa_backend", "cudnn_fe",
                    "--num_warmup_iterations", str(warmup_repeats),
                    "--num_iterations", str(repeats),
                    "--fwd_bwd",
                    "--format_output",  # Get CSV-formatted output
                    "--skip_ref",
                ]
                if varlen:
                    cmd.append("--var_len")
                if deterministic:
                    cmd.append("--deterministic")
                if window_size[0] is not None:
                    cmd.append("--window_size")
                    cmd.append(str(window_size[0]))
                print('\tbenchmark_single_sdpa.py:')
                try:
                    cudnn_result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        check=False,  # Don't raise exception on non-zero exit
                    )
                    output_line = cudnn_result.stdout.strip().split('\n')[-1]
                    parts = output_line.split(',')
                    fwd_time = parts[8]
                    bwd_time = parts[9]
                    fwd_flops = parts[10]
                    bwd_flops = parts[11]
                    print(f'\t\tForward: {fwd_time} ms, {fwd_flops} TFLOPS')
                    print(f'\t\tBackward: {bwd_time} ms, {bwd_flops} TFLOPS')
                    output_config = f'{batch_size},{nheads},{nheads_kv},{seqlen},{headdim},{window_size[0]},{causal},{varlen},{deterministic}'
                    output_file.write(f'{output_config},benchmark_single_sdpa,{fwd_time},{fwd_flops},')
                    output_file.write(f',{bwd_time},{bwd_flops},')
                    output_file.write('\n')
                    output_file.flush()
                except Exception as e:
                    print(f"  Failed to run benchmark_single_sdpa.py\n    cmd: {' '.join(cmd)}\n    stdout: {cudnn_result.stdout}\n    stderr: {cudnn_result.stderr}")
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--no_run_fa4", action="store_true")
    parser.add_argument("--run_cudnn", action="store_true")
    parser.add_argument("--dropout_p", type=float, default=0.0)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp8"])
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--varlen", action="store_true")
    parser.add_argument("--no_backward", action="store_true")
    parser.add_argument("--page_size", type=int, default=None)
    parser.add_argument("--softcap", type=float, default=0.0)
    parser.add_argument("--V_colmajor", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seqlen", type=int, default=8192)
    parser.add_argument("--nheads", type=int, default=16)
    parser.add_argument("--nheads_kv", type=int, default=16)
    parser.add_argument("--headdim", type=int, default=128)
    parser.add_argument("--window_size", type=int, default=None)
    args = parser.parse_args()
    print(' start benchmark '.center(100, '='))
    if args.sweep:
        sweep(args.output_file, not args.no_run_fa4, args.run_cudnn)
    else:
        benchmark_attn(
            run_fa4 = not args.no_run_fa4,
            run_cudnn = args.run_cudnn,
            dropout_p = args.dropout_p,
            causal = args.causal,
            dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float8_e4m3fn,
            verbose = args.verbose,
            varlen = args.varlen,
            has_backward = not args.no_backward,
            page_size = args.page_size if args.page_size is not None else None,
            softcap = args.softcap,
            V_colmajor = args.V_colmajor,
            deterministic = args.deterministic,
            batch_size = args.batch_size,
            seqlen = args.seqlen,
            nheads = args.nheads,
            nheads_kv = args.nheads_kv,
            headdim = args.headdim,
            window_size = (args.window_size, args.window_size),
        )
    print(' end benchmark '.center(100, '='))
