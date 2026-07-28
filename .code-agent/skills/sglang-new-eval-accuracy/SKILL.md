---
name: sglang-new-eval-accuracy
description: Run the fixed WeLMv4/SGLang 80b_v4d5 new_eval accuracy regression and report rollout speed metrics against a running OpenAI-compatible SGLang endpoint, with optional dataset selection. Use when Codex is asked to run new_eval, run the 80b_v4d5 precision/accuracy and speed backtest for all or selected datasets, compare accuracy against the bundled TP4-DP16 baseline, or produce a pass/fail accuracy report with per-dataset throughput and latency.
---

# SGLang New Eval Accuracy and Speed

Run the fixed `80b_v4d5` `new_eval` accuracy regression, report rollout throughput and latency, and produce a required accuracy `PASS`/`FAIL` conclusion. Prefer the bundled script over retyping shell/Python snippets.

The expected user request is a single sentence like: `SGLang 推理服务运行在 http://127.0.0.1:8000，使用 new_eval 跑 80b_v4d5 精度和速度回测，并发设置为 36，数据集用 aa-lcr。`

## Fixed Contract

Required input:

- SGLang OpenAI-compatible endpoint, for example `http://127.0.0.1:8000`.

Optional input:

- Concurrency. Default to `20` if omitted. Use the user's value for all selected jobs and the judge concurrency.
- Dataset selection. Default to all supported datasets if omitted. If the user specifies a dataset, run only the requested dataset(s), for example `数据集用 gpqa-diamond`.

Extract only these user-controlled values:

- Serving URL, for example `http://127.0.0.1:8000`.
- Concurrency, for example `36`.
- Dataset selection, for example `aa-lcr`. If omitted, run all supported datasets.

Do not ask for the binary URL, supported dataset list, output directory, judge config, judge API key, or baseline. Use these fixed values:

```text
new_eval binary = https://mirrors.tencent.com/repository/generic/welm/new_eval/bin/20260728/new_eval-linux-amd64-efa9b634
new_eval sha256 = efa9b634a81a379e1cef8760d4c72b032a1ee3fd1e6d02834d3b18b146f43b95
new_eval commit = 39cdc0e76de51f79d70a63a6bbb4c7f9fb427c93
baseline_dir    = <this skill>/assets/baselines/80b_v4d5/20260607/TP4-DP16
judge_config    = <this skill>/assets/configs/eval.example.5tasks.yaml
run_root        = /tmp/new_eval/80b_v4d5
score_tolerance = 0.01
```

Do not use host-specific absolute home paths. The baseline metrics and judge config are bundled inside this skill and must be read from the skill directory containing this `SKILL.md`.

The judge API key is read from the bundled judge config. Do not print the key, dump the full generated `config.yaml`, or include secrets in the final response.

Interpret `score_tolerance = 0.01` as an absolute 1 percentage point drop:

```text
task passes score gate iff actual_score >= baseline_score - 0.01
```

Read `score` and `sample_count` from the bundled `*_metrics.json` files at run time. To update the baseline, replace the JSON files under `assets/baselines/80b_v4d5/20260607/TP4-DP16/`.

## Dataset Selection

Supported datasets are fixed. Run all of them by default, in this order:

| job name | task type | copied metrics file |
|---|---|---|
| `suite_gary_math` | `gary-math` | `gary_math_metrics.json` |
| `suite_aime_2025` | `aime-2025` | `aime_2025_metrics.json` |
| `suite_gpqa_diamond` | `gpqa-diamond` | `gpqa_diamond_metrics.json` |
| `suite_aa_lcr` | `aa-lcr` | `aa_lcr_metrics.json` |
| `suite_aa_omniscience` | `aa-omniscience-public` | `aa_omniscience_metrics.json` |

If the user requests one or more datasets, pass only those task types to `--tasks`. Preserve the fixed order above. Accept comma-separated, Chinese-comma-separated, or whitespace-separated values. If any requested dataset is unsupported, do not run `new_eval`; report the supported task types.

## Run Command

Locate this skill folder from the path of the `SKILL.md` you loaded, then run its bundled script:

```bash
/path/to/sglang-new-eval-accuracy/scripts/run_new_eval_accuracy.sh \
  --base-url "http://127.0.0.1:8000" \
  --concurrency 20 \
  --tasks "gpqa-diamond"
```

Omit `--tasks` to run all datasets. Replace `/path/to/sglang-new-eval-accuracy` with the actual skill directory; the script then resolves `assets/` relative to itself. Do not substitute a fixed user home path.

For local skill validation only, use `--prepare-only` to generate config and selected task files without downloading or running `new_eval`.

## Run Directory

The script creates one timestamped run directory per invocation:

```text
${RUN_DIR}/
  config.yaml
  command.txt
  run.log
  new_eval.exitcode
  models.json
  selected_tasks.json
  binary/
    new_eval
    new_eval.sha256
    new_eval.url.txt
    new_eval.git-commit.txt
    new_eval.help.txt
  baseline/
    *_metrics.json
  outputs/
  metrics/
    *_metrics.json
  summary.json
  summary.csv
  summary.md
```

The script copies bundled baseline JSONs into `${RUN_DIR}/baseline/` so each run is self-contained.

## Workflow

1. Normalize the user URL to an OpenAI API base URL. If the user gives `http://host:port`, the script uses `http://host:port/v1`. If the URL already ends in `/v1`, it keeps it.
2. Probe `${API_BASE}/models` with `curl --noproxy '*'`. The script saves the response to `${RUN_DIR}/models.json` and uses the first model id if available; otherwise it uses `welmv4`.
3. Download the fixed `new_eval` binary into `${RUN_DIR}/binary/new_eval`, verify its pinned SHA256, save its URL and Git commit, `chmod +x` it, and save `new_eval --help`.
4. Generate `${RUN_DIR}/config.yaml` and `${RUN_DIR}/selected_tasks.json` from the bundled judge config without printing the judge API key.
5. Run `new_eval run --config "${RUN_DIR}/config.yaml"` with proxy variables unset and tee all output to `${RUN_DIR}/run.log`.
6. Copy each task's produced `metrics.json` into `${RUN_DIR}/metrics/<copied metrics file>`.
7. Generate `summary.json`, `summary.csv`, and `summary.md` with accuracy, throughput, and latency fields, even when `new_eval` exits non-zero.
8. In the final answer, show the first line conclusion from `summary.md`, the accuracy and speed tables, and the run directory path.

## Pass/Fail Rules

Overall `PASS` requires all conditions:

- `new_eval` exit code is `0`.
- All selected actual metrics files exist.
- All selected baseline metrics files exist in `${RUN_DIR}/baseline/`, copied from this skill's bundled baseline assets.
- For every selected task, `actual.sample_count == baseline.sample_count`.
- For every selected task, `empty_predict_count == 0`.
- For every selected task, `judge_parse_failed_count == 0`.
- For every selected task, `actual.score >= baseline.score - 0.01`.

Any missing file, non-zero exit code, sample count mismatch, failed sample, or score drop beyond 1 percentage point is `FAIL`.

Speed metrics are informational and do not affect `PASS`/`FAIL`. Report them per dataset from `metrics.bench_metrics`. Use total request/token throughput only; do not request `gpu_count` or report per-GPU throughput. TTFT, ITL, E2E, and throughput cover rollout activity only and exclude judge time.

## Final Response

Always include:

- The overall `PASS`/`FAIL` conclusion.
- The path to `${RUN_DIR}`.
- The `new_eval` artifact filename, Git commit, and SHA256 from `summary.md`.
- The selected task list.
- The score table from `summary.md`.
- Both speed tables from `summary.md`: throughput and latency.
- Any speed notes, including missing `metrics.bench_metrics`.
- A short list of fail reasons when the result is `FAIL`.

Do not print the judge API key or the full generated `config.yaml` in the final response.
