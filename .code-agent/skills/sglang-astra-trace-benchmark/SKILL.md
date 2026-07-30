---
name: sglang-astra-trace-benchmark
description: Replay a pinned Astra production chat-completion trace against an already running SGLang OpenAI-compatible endpoint and report serving metrics. Use when a user asks to benchmark or 压测 SGLang with 线上/生产 trace, with either the full trace by default or an explicit total request count/总数据量.
---

# SGLang Astra Trace Benchmark

Run the bundled wrapper. Follow the same external-source pattern as `sglang-new-eval-accuracy`: keep benchmark code and trace outside the SGLang repository, download one pinned binary artifact containing the production trace, verify its checksum, then execute.

## Inputs

Require only:

- SGLang endpoint URL

Accept optional overrides:

- Concurrency (default: 8)
- Total measured request count (default: all replayable trace rows)
- Served model name
- Custom trace path
- Run root

Map common user language as follows:

- `sglang运行在 http://host:port` -> `--base-url http://host:port`
- `使用线上的trace` -> omit `--trace` and use the production trace embedded in the pinned binary
- `并发为32` -> `--concurrency 32`
- `总数据量为128` -> `--sample-num 128`
- An explicit model -> `--model MODEL`; otherwise use the first id from `/v1/models`, falling back to `welmv4`

The replay is conversation-serial closed loop. It preserves turn order inside each real conversation while running different conversations concurrently. Omit `--sample-num` to replay every usable row in the trace; `--sample-num N` sends exactly N measured requests.

## Pinned Sources

```text
binary_url          = https://mirrors.tencent.com/repository/generic/welm/astra_trace_bench/bin/20260730/astra-trace-bench-linux-amd64-690b8fa6
binary_sha256       = 690b8fa60a60663e0f40a2d59af8f81ec54897c5073730423d51a2b2778aed29
source_commit       = 7dc430d324770da606a81ce042a0a15fad7dbee4
embedded_trace_file = welm_2026-07-12_10-00_1000_no_decode1.jsonl
embedded_trace_sha  = 2e93cac34d6e8d2ae0f89769465cb6bf9048fd2d6dd6c332123cf2adf693106a

run_root      = /tmp/astra_trace_bench
```

The embedded trace has 1000 rows and excludes requests whose recorded decode length is 1. The benchmark loader skips rows without a usable recorded `completion_tokens`; full-trace mode reports the resulting replayable count. Do not print or summarize prompt contents. The default path requires neither WOA Git access nor Git LFS.

## Run Command

Locate this skill folder from the loaded `SKILL.md`, then run:

```bash
/path/to/sglang-astra-trace-benchmark/scripts/run_astra_trace_benchmark.sh \
  --base-url "http://127.0.0.1:30000" \
  --concurrency 32
```

Optional arguments:

- `--sample-num N` to limit the run to exactly `N` measured requests; omit it to run the full trace
- `--trace PATH` to use a local custom JSONL or JSONL.GZ instead of the embedded trace
- `--model MODEL` to skip model auto-detection
- `--run-root DIR` to override the result root
- `--prepare-only` to generate metadata and `command.txt` without downloading or running

If authentication is required, set `SGLANG_API_KEY`. Never print or copy its value into logs or the final response.

## Workflow

1. Normalize a bare server URL, `/v1` URL, or full `/v1/chat/completions` URL.
2. Probe `${API_BASE}/models` and select the first model id unless the user supplied one.
3. Download the fixed Linux/amd64 binary from the internal artifact repository with proxy variables cleared, verify its pinned SHA256, and refuse to run on a mismatch.
4. Unless the user supplied `--trace`, omit the benchmark's `--input` flag so it loads the production trace embedded in the binary.
5. Run the default conversation-serial replay with the requested concurrency, using the full trace unless the user supplied a sample count.
6. Generate `summary.json` and `summary.md`, even when the benchmark reports request failures.

## Run Directory

```text
${RUN_DIR}/
  command.txt
  models.json
  run.log
  bench.exitcode
  bench_metrics.json
  requests.jsonl
  summary.json
  summary.md
  binary/
    astra-trace-bench
    astra-trace-bench.sha256
    astra-trace-bench.url.txt
    astra-trace-bench.git-commit.txt
    astra-trace-bench.help.txt
  input/
    trace.source.txt
    trace.sha256
```

## Pass/Fail Rules

Overall `PASS` requires all conditions:

- Benchmark exit code is 0.
- With `--sample-num N`, `sample_count == N`; without it, `sample_count > 0` after the loader has exhausted the trace.
- `successful_requests == sample_count`.
- `failed_requests == 0`.
- `empty_predict_count == 0`.
- `requests.jsonl` contains exactly `sample_count` records.

Treat any mismatch as `FAIL`.

## Final Response

Always include:

- Overall `PASS` or `FAIL`
- Run directory
- Binary artifact URL and SHA256
- Source Git commit
- Embedded trace filename and SHA256, or the custom trace path
- Model, configured concurrency, and observed max concurrency
- Sample/success/failure counts
- Request rate, output token throughput, TTFT, ITL, and E2E metrics
- Failure reasons when present
