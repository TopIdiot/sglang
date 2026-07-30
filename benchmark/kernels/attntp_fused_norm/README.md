# AttnTP Fused Norm Benchmark

This benchmark compares the production fused AttnTP norm backends with the
equivalent NCCL AllReduce plus WeLM norm path. It covers prefill and decode,
hidden sizes 2048 and 4096, eager execution, and CUDA graph replay.

Run from the repository root with one process per AttnTP rank:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.run \
  --standalone \
  --nproc-per-node=2 \
  benchmark/kernels/attntp_fused_norm/benchmark.py \
  --phase both \
  --hidden-sizes 2048 4096 \
  --backends ipc symm
```

Use `--prefill-rows` and `--decode-rows` to restrict the shape sweep. Select
`--topology tp`, `dp`, or `cp` to match the production ownership contract,
select `--execution eager` or `--execution graph`, and use `--output PATH` to
retain the JSON results.

The script validates output and residual error before reporting latency. A
failed correctness guard invalidates the corresponding timing result.

The replicated baseline always uses the NCCL device process group for
AllReduce, followed by `mmq_style_norm_after_attn`. Timing uses these rules:

- Warmup completes before samples are collected.
- A persistent 256 MiB buffer is swept before every measured replay to evict
  prior operands from L2. The sweep is ordered on the benchmark stream but is
  outside the CUDA event interval.
- All replay/event pairs are enqueued before a single CUDA synchronization.
- Timed fused Prefill calls include the `actual_rows` update. Prefill CP also
  includes the lane-rotation update; Prefill TP/DP and all Decode calls do not.
- Per-sample latency is reduced with MAX across ranks, then summarized with
  median, P90, minimum, and maximum.
