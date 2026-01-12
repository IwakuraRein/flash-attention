GPU under benchmark: computelab `NVIDIA B200`
Power lock (max allowed is 1000): `sudo nvidia-smi -pl 1200`
Frequency lock: `sudo nvidia-smi -ac 3996,1965`

Run benchmark_single_sdpa.py
```
python benchmark_single_sdpa.py \
    --batch_size 1 \
    --head_dim 128 \
    --num_q_heads 128 \
    --num_kv_heads 16 \
    --q_seqlen 512 \
    --kv_seqlen 512 \
    --attn_mask top_left \
    --data_type bfloat16 \
    --sdpa_backend cudnn_fe \
    --num_warmup_iterations 5 \
    --num_iterations 10 \
    --fwd_bwd \
    --skip_ref \
    --verbose
```

Run nsys to compare profiled kernel times
```
nsys profile \
    --stats=true \
    --force-overwrite=true \
    -o nsys_results \
    python benchmark_single_sdpa.py \
    --batch_size 1 \
    --head_dim 128 \
    --num_q_heads 128 \
    --num_kv_heads 16 \
    --q_seqlen 512 \
    --kv_seqlen 512 \
    --attn_mask top_left \
    --data_type bfloat16 \
    --sdpa_backend cudnn_fe \
    --num_warmup_iterations 5 \
    --num_iterations 10 \
    --fwd_bwd \
    --skip_ref
```

Run with ncu to compare profiled kernel times
```
ncu --print-summary per-kernel \
    --page details \
    --metrics gpu__time_duration.sum \
    python benchmark_single_sdpa.py \
    --batch_size 1 \
    --head_dim 128 \
    --num_q_heads 128 \
    --num_kv_heads 16 \
    --q_seqlen 512 \
    --kv_seqlen 512 \
    --attn_mask top_left \
    --data_type bfloat16 \
    --sdpa_backend cudnn_fe \
    --num_warmup_iterations 0 \
    --num_iterations 1 \
    --fwd_bwd \
    --skip_ref
```


# Torch profiler:
## Forward kernels
* cudnn_generated_fort_native_sdpa_sm100_flash_fprop_f16_knob_7_128x128x128_4x1x1_cga1x1x1_kernel0_0 0.023ms
Total: 0.023ms

## Backward kernels:
* cudnn::fusion::compute_dot_do_o_specialized<true, 128> 0.014ms
* cudnn_generated_fort_native_sdpa_sm100_flash_bprop_f16_knob_31_128x128x128_1x4x1_cga1x1x1_kernel0_0 0.039 ms
* udnn::fusion::convert_dq_to_16bits<true> 0.008ms
* cudnn::fusion::fmha_reduce_head<true> 0.005ms
Total: 0.066ms

# nsys profiler
## Kernels
* vectorized_elementwise_kernel 69.3us
* distribution_elementwise_grid_stride_kernel 12.6us
* cudnn_generated_fort_native_sdpa_sm100_flash_bprop_f16_knob_31_128x128x128_1x4x1_cga1x1x1_kernel0_0 38.9us
* cudnn_generated_fort_native_sdpa_sm100_flash_fprop_f16_knob_7_128x128x128_4x1x1_cga1x1x1_kernel0_0 23.3us
* compute_dot_do_o_specialized 14.2us
* fmha_reduce_head 5.0us
* convert_dq_to_16bits 7.8us
* distribution_elementwise_grid_stride_kernel 2.4us

# ncu profiler

## Kernels
* cudnn_generated_fort_native_sdpa_sm100_flash_bprop_f16_knob_31_128x128x128_1x4x1_cga1x1x1_kernel0_0 (128, 1, 4)x(512, 1, 1) 68.7us (called 1 times)
* cudnn_generated_fort_native_sdpa_sm100_flash_fprop_f16_knob_7_128x128x128_4x1x1_cga1x1x1_kernel0_0 41.0us (called 1 times)
* distribution_elementwise_grid_stride_kernel 23.65us (called 8 times)
* distribution_elementwise_grid_stride_kernel 5.65us (called 2 times)
* vectorized_elementwise_kernel 112.8us (called 2 times)
* cudnn::compute_dot_do_o_specialized 18.1us (called 1 times)
* cudnn::convert_dq_to_16bits 12.93 us (called 1 times)
* cudnn::fmha_reduce_head<1> 9.0us (called 2 times)
Total: 0.58ms

# side by side comparison

## torch vs ncu
| Kernel Name | Torch Profiler (ms) | NCU (ms) | Difference | Ratio (NCU/Torch) |
|-------------|--------------------:|----------:|-----------:|------------------:|
| **Forward Pass** |
| `cudnn_generated_fort_native_sdpa_sm100_flash_fprop_f16_knob_7_..._kernel0_0` | 0.023 | 0.041 | +0.018 | 1.78× |
| **Backward Pass** |
| `cudnn_generated_fort_native_sdpa_sm100_flash_bprop_f16_knob_31_..._kernel0_0` | 0.039 | 0.069 | +0.030 | 1.77× |
| `compute_dot_do_o_specialized` | 0.014 | 0.018 | +0.004 | 1.29× |
| `convert_dq_to_16bits` | 0.008 | 0.013 | +0.005 | 1.62× |
| `fmha_reduce_head` | 0.005 | 0.009* | +0.004 | 1.80× |

## torch vs nsys
| Kernel Name | Torch Profiler (ms) | Nsys (ms) | Difference (ms) | Ratio (Nsys/Torch) |
|-------------|--------------------:|----------:|----------------:|-------------------:|
| **Forward Pass** |
| `cudnn_generated_fort_native_sdpa_sm100_flash_fprop_f16_knob_7_..._kernel0_0` | 0.023 | 0.023 | 0.000 | 1.01× |
| **Backward Pass** |
| `cudnn_generated_fort_native_sdpa_sm100_flash_bprop_f16_knob_31_..._kernel0_0` | 0.039 | 0.039 | 0.000 | 1.00× |
| `compute_dot_do_o_specialized` | 0.014 | 0.014 | 0.000 | 1.01× |
| `convert_dq_to_16bits` | 0.008 | 0.008 | 0.000 | 0.98× |
| `fmha_reduce_head` | 0.005 | 0.005 | 0.000 | 1.00× |
