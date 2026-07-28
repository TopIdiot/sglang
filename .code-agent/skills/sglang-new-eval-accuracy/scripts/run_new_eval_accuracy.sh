#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_new_eval_accuracy.sh --base-url URL [--concurrency N] [--tasks LIST] [--run-root DIR]
  run_new_eval_accuracy.sh --help

Required:
  --base-url URL       SGLang OpenAI-compatible endpoint. A trailing /v1 is optional.

Optional:
  --concurrency N      Concurrency for selected jobs and judge. Default: 20.
  --tasks LIST         Dataset list. Empty means all datasets.
                       Accepts comma, Chinese-comma, or whitespace separators.
  --run-root DIR       Output root. Default: /tmp/new_eval/80b_v4d5.
  --prepare-only       Generate run directory/config and stop before downloading/running new_eval.

EOF
}

BASE_URL_RAW=""
CONCURRENCY="20"
SELECTED_TASKS_RAW=""
RUN_ROOT="${NEW_EVAL_RUN_ROOT:-/tmp/new_eval/80b_v4d5}"
PREPARE_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-url)
      BASE_URL_RAW="${2:-}"
      shift 2
      ;;
    --concurrency)
      CONCURRENCY="${2:-}"
      shift 2
      ;;
    --tasks)
      SELECTED_TASKS_RAW="${2:-}"
      shift 2
      ;;
    --run-root)
      RUN_ROOT="${2:-}"
      shift 2
      ;;
    --prepare-only)
      PREPARE_ONLY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'unknown argument: %s\n\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${BASE_URL_RAW}" ]]; then
  printf 'missing required --base-url\n\n' >&2
  usage >&2
  exit 2
fi

if [[ ! "${CONCURRENCY}" =~ ^[0-9]+$ ]] || [[ "${CONCURRENCY}" -lt 1 ]]; then
  printf 'invalid --concurrency: %s\n' "${CONCURRENCY}" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
SKILL_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
BASELINE_SOURCE_DIR="${SKILL_DIR}/assets/baselines/80b_v4d5/20260607/TP4-DP16"
JUDGE_CONFIG="${SKILL_DIR}/assets/configs/eval.example.5tasks.yaml"

if [[ ! -d "${BASELINE_SOURCE_DIR}" ]]; then
  printf 'missing bundled baseline dir: %s\n' "${BASELINE_SOURCE_DIR}" >&2
  exit 2
fi

if [[ ! -f "${JUDGE_CONFIG}" ]]; then
  printf 'missing bundled judge config: %s\n' "${JUDGE_CONFIG}" >&2
  exit 2
fi

NEW_EVAL_URL="https://mirrors.tencent.com/repository/generic/welm/new_eval/bin/20260728/new_eval-linux-amd64-efa9b634"
NEW_EVAL_SHA256="efa9b634a81a379e1cef8760d4c72b032a1ee3fd1e6d02834d3b18b146f43b95"
NEW_EVAL_GIT_COMMIT="39cdc0e76de51f79d70a63a6bbb4c7f9fb427c93"

RUN_ID="$(date -u '+%Y%m%d_%H%M%S')_pid$$"
RUN_DIR="${RUN_ROOT%/}/${RUN_ID}"
BIN_DIR="${RUN_DIR}/binary"
BIN="${BIN_DIR}/new_eval"

mkdir -p "${BIN_DIR}" "${RUN_DIR}/outputs" "${RUN_DIR}/metrics" "${RUN_DIR}/baseline"

BASE_URL="${BASE_URL_RAW%/}"
case "${BASE_URL}" in
  */v1) API_BASE="${BASE_URL}" ;;
  *) API_BASE="${BASE_URL}/v1" ;;
esac

cp "${BASELINE_SOURCE_DIR}"/*_metrics.json "${RUN_DIR}/baseline/"

if [[ "${PREPARE_ONLY}" -eq 0 ]] && curl --noproxy '*' -fsS "${API_BASE}/models" > "${RUN_DIR}/models.json"; then
  MODEL_ID="$(python3 -c 'import json,sys; d=json.load(sys.stdin); print((d.get("data") or [{}])[0].get("id",""))' < "${RUN_DIR}/models.json" || true)"
else
  printf '{}\n' > "${RUN_DIR}/models.json"
  MODEL_ID=""
fi
MODEL_ID="${MODEL_ID:-welmv4}"

export RUN_DIR API_BASE MODEL_ID CONCURRENCY SELECTED_TASKS_RAW JUDGE_CONFIG
export NEW_EVAL_URL NEW_EVAL_SHA256 NEW_EVAL_GIT_COMMIT
python3 - <<'PY'
import json
import os
import re
from pathlib import Path

TASKS = [
    {"task": "gary-math", "job": "suite_gary_math", "metrics": "gary_math_metrics.json", "aliases": ["gary_math"]},
    {"task": "aime-2025", "job": "suite_aime_2025", "metrics": "aime_2025_metrics.json", "aliases": ["aime_2025"]},
    {"task": "gpqa-diamond", "job": "suite_gpqa_diamond", "metrics": "gpqa_diamond_metrics.json", "aliases": ["gpqa_diamond"]},
    {"task": "aa-lcr", "job": "suite_aa_lcr", "metrics": "aa_lcr_metrics.json", "aliases": ["aa_lcr"]},
    {"task": "aa-omniscience-public", "job": "suite_aa_omniscience", "metrics": "aa_omniscience_metrics.json", "aliases": ["aa-omniscience", "aa_omniscience"]},
]

ALIASES = {}
for item in TASKS:
    names = [item["task"], item["job"], item["metrics"], item["metrics"].replace("_metrics.json", "")]
    names.extend(item.get("aliases", []))
    for name in names:
        ALIASES[name.lower()] = item["task"]
        ALIASES[name.replace("_", "-").lower()] = item["task"]

def parse_selected(raw):
    names = [part.strip() for part in re.split(r"[\s,，]+", raw) if part.strip()]
    if not names:
        return TASKS
    seen = set()
    unknown = []
    for name in names:
        key = name.lower()
        task = ALIASES.get(key) or ALIASES.get(key.replace("_", "-"))
        if not task:
            unknown.append(name)
            continue
        if task in seen:
            continue
        seen.add(task)
    if unknown:
        supported = ", ".join(item["task"] for item in TASKS)
        raise SystemExit(f"unsupported dataset(s): {', '.join(unknown)}; supported: {supported}")
    return [item for item in TASKS if item["task"] in seen]

def y(value):
    return json.dumps(str(value), ensure_ascii=False)

def read_judge_config(path):
    text = Path(path).read_text()
    m = re.search(r"(?ms)^judge:\n(?P<body>(?:^[ \t]+.*\n?)+)", text)
    if not m:
        raise SystemExit(f"missing judge block in bundled config: {path}")
    judge = m.group("body")

    def field(name, default=None):
        mm = re.search(rf"(?m)^[ \t]+{re.escape(name)}:[ \t]*(.*?)[ \t]*(?:#.*)?$", judge)
        if not mm:
            if default is None:
                raise SystemExit(f"missing judge.{name} in bundled config: {path}")
            return default
        value = mm.group(1).strip()
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        return value

    return {
        "base_url": field("base_url", "http://welmgateway.production.polaris:30000/v1"),
        "api_key": field("api_key"),
        "model": field("model", "gpt-oss-120b-eval"),
        "rtx": field("rtx", "wenhanli_eval"),
        "max_retries": int(field("max_retries", "5")),
    }

selected_tasks = [
    {"task": item["task"], "job": item["job"], "metrics": item["metrics"]}
    for item in parse_selected(os.environ.get("SELECTED_TASKS_RAW", ""))
]
run_dir = Path(os.environ["RUN_DIR"])
(run_dir / "selected_tasks.json").write_text(json.dumps(selected_tasks, indent=2, ensure_ascii=False))
judge_config = read_judge_config(os.environ["JUDGE_CONFIG"])

job_lines = []
for item in selected_tasks:
    job_lines.append(f"""  - name: {item['job']}
    type: {item['task']}
    concurrency: {int(os.environ["CONCURRENCY"])}
    rollout_max_retries: 2
""")

config = f"""version: v1

defaults:
  base_url: {y(os.environ["API_BASE"])}
  api_key: "dummy"
  model: {y(os.environ["MODEL_ID"])}
  rtx: ""
  output_root: {y(str(run_dir / "outputs"))}

judge:
  base_url: {y(judge_config["base_url"])}
  api_key: {y(judge_config["api_key"])}
  model: {y(judge_config["model"])}
  rtx: {y(judge_config["rtx"])}
  concurrency: {int(os.environ["CONCURRENCY"])}
  max_retries: {int(judge_config["max_retries"])}

jobs:
{''.join(job_lines)}"""

(run_dir / "config.yaml").write_text(config)
PY

if [[ "${PREPARE_ONLY}" -eq 1 ]]; then
  printf 'Prepared run dir: %s\n' "${RUN_DIR}"
  exit 0
fi

curl -L --fail --retry 3 --noproxy '*' -o "${BIN}" "${NEW_EVAL_URL}"
printf '%s  %s\n' "${NEW_EVAL_SHA256}" "${BIN}" | sha256sum --check
chmod +x "${BIN}"
sha256sum "${BIN}" | tee "${BIN_DIR}/new_eval.sha256"
printf '%s\n' "${NEW_EVAL_URL}" > "${BIN_DIR}/new_eval.url.txt"
printf '%s\n' "${NEW_EVAL_GIT_COMMIT}" > "${BIN_DIR}/new_eval.git-commit.txt"
"${BIN}" --help > "${BIN_DIR}/new_eval.help.txt"

RUN_ARGS=(run --config "${RUN_DIR}/config.yaml")

printf 'env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY NO_PROXY=* no_proxy=* stdbuf -oL %q' "${BIN}" > "${RUN_DIR}/command.txt"
printf ' %q' "${RUN_ARGS[@]}" >> "${RUN_DIR}/command.txt"
printf '\n' >> "${RUN_DIR}/command.txt"

set +e
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
  NO_PROXY='*' no_proxy='*' \
  stdbuf -oL "${BIN}" "${RUN_ARGS[@]}" 2>&1 | tee "${RUN_DIR}/run.log"
NEW_EVAL_EXIT="${PIPESTATUS[0]}"
set -e
printf '%s\n' "${NEW_EVAL_EXIT}" > "${RUN_DIR}/new_eval.exitcode"

export RUN_DIR
python3 - <<'PY'
import json
import os
import shutil
from pathlib import Path

RUN_DIR = Path(os.environ["RUN_DIR"])
for item in json.loads((RUN_DIR / "selected_tasks.json").read_text()):
    src = RUN_DIR / "outputs" / item["job"] / "output" / item["task"] / "metrics.json"
    dst = RUN_DIR / "metrics" / item["metrics"]
    if src.exists():
        shutil.copy2(src, dst)
PY

export RUN_DIR API_BASE MODEL_ID CONCURRENCY BASELINE_SOURCE_DIR
export NEW_EVAL_URL NEW_EVAL_SHA256 NEW_EVAL_GIT_COMMIT
python3 - <<'PY'
import csv
import json
import os
from pathlib import Path

RUN_DIR = Path(os.environ["RUN_DIR"])
BASELINE_DIR = RUN_DIR / "baseline"
BASELINE_SOURCE_DIR = Path(os.environ["BASELINE_SOURCE_DIR"])
TOL = 0.01

TASKS = [
    (item["task"], item["job"], item["metrics"])
    for item in json.loads((RUN_DIR / "selected_tasks.json").read_text())
]

def load_metrics(path):
    with path.open() as f:
        data = json.load(f)
    return data.get("metrics", data)

def as_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default

def as_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default

def as_optional_int(value):
    try:
        return int(value)
    except Exception:
        return None

def as_optional_float(value):
    try:
        return float(value)
    except Exception:
        return None

def format_speed_float(value):
    return "N/A" if value is None else f"{value:.2f}"

def format_speed_int(value):
    return "N/A" if value is None else str(value)

def count_correct_from_output(job, task, sample_count, score):
    output_path = RUN_DIR / "outputs" / job / "output" / task / f"{task}_output.json"
    if not output_path.exists():
        return round(score * sample_count)
    try:
        data = json.loads(output_path.read_text())
    except Exception:
        return round(score * sample_count)
    samples = data.get("batch_samples") or data.get("samples") or []
    correct = 0
    seen = 0
    for sample in samples:
        metric = ((sample.get("meta") or {}).get("metric") or {})
        if "correct" in metric:
            seen += 1
            correct += bool(metric.get("correct"))
        elif "score" in sample:
            seen += 1
            correct += as_float(sample.get("score")) >= 1.0
    if seen == sample_count:
        return int(correct)
    return round(score * sample_count)

exit_path = RUN_DIR / "new_eval.exitcode"
try:
    exit_code = int(exit_path.read_text().strip())
except Exception:
    exit_code = 1

rows = []
fail_reasons = []
speed_notes = []
if exit_code != 0:
    fail_reasons.append(f"new_eval exited with code {exit_code}")

for task, job, metrics_name in TASKS:
    task_reasons = []
    baseline_path = BASELINE_DIR / metrics_name
    actual_path = RUN_DIR / "metrics" / metrics_name

    if not baseline_path.exists():
        task_reasons.append(f"missing baseline metrics: {baseline_path}")
        baseline = {}
    else:
        baseline = load_metrics(baseline_path)

    if not actual_path.exists():
        task_reasons.append(f"missing actual metrics: {actual_path}")
        actual = {}
    else:
        actual = load_metrics(actual_path)

    bench_metrics = actual.get("bench_metrics")
    if not isinstance(bench_metrics, dict) or not bench_metrics:
        bench_metrics = {}
        if actual_path.exists():
            speed_notes.append(f"{task}: missing metrics.bench_metrics")

    baseline_score = as_float(baseline.get("score"))
    baseline_samples = as_int(baseline.get("sample_count"))
    score = as_float(actual.get("score"))
    sample_count = as_int(actual.get("sample_count"))
    empty = as_int(actual.get("empty_predict_count"))
    judge_fail = as_int(actual.get("judge_parse_failed_count"))
    failed = empty + judge_fail
    expected_samples = baseline_samples

    if expected_samples and sample_count != expected_samples:
        task_reasons.append(f"sample_count {sample_count} != expected {expected_samples}")
    if empty > 0:
        task_reasons.append(f"empty_predict_count={empty}")
    if judge_fail > 0:
        task_reasons.append(f"judge_parse_failed_count={judge_fail}")
    if actual_path.exists() and baseline_path.exists() and score < baseline_score - TOL:
        task_reasons.append(f"score drop {score - baseline_score:+.4f} exceeds -{TOL:.4f}")

    correct = count_correct_from_output(job, task, sample_count, score) if actual_path.exists() else 0
    wrong = max(sample_count - correct - failed, 0)
    status = "PASS" if not task_reasons else "FAIL"
    if task_reasons:
        fail_reasons.extend([f"{task}: {reason}" for reason in task_reasons])

    rows.append({
        "task": task,
        "status": status,
        "score": score,
        "baseline_score": baseline_score,
        "delta": score - baseline_score,
        "threshold": baseline_score - TOL,
        "sample_count": sample_count,
        "expected_sample_count": expected_samples,
        "baseline_sample_count": baseline_samples,
        "correct_count": correct,
        "wrong_count": wrong,
        "failed_count": failed,
        "empty_predict_count": empty,
        "judge_parse_failed_count": judge_fail,
        "speed_metrics_available": bool(bench_metrics),
        "duration_scope": str(bench_metrics.get("duration_scope") or ""),
        "duration_sec": as_optional_float(bench_metrics.get("duration_sec")),
        "successful_requests": as_optional_int(bench_metrics.get("successful_requests")),
        "failed_requests": as_optional_int(bench_metrics.get("failed_requests")),
        "total_prompt_tokens": as_optional_int(bench_metrics.get("total_prompt_tokens")),
        "total_completion_tokens": as_optional_int(bench_metrics.get("total_completion_tokens")),
        "request_rate_per_s": as_optional_float(bench_metrics.get("request_rate_per_s")),
        "prompt_tokens_per_s": as_optional_float(bench_metrics.get("prompt_tokens_per_s")),
        "output_tokens_per_s": as_optional_float(bench_metrics.get("output_tokens_per_s")),
        "completion_tokens_avg": as_optional_float(bench_metrics.get("completion_tokens_avg")),
        "ttft_avg_ms": as_optional_float(bench_metrics.get("ttft_avg_ms")),
        "ttft_p50_ms": as_optional_float(bench_metrics.get("ttft_p50_ms")),
        "ttft_p99_ms": as_optional_float(bench_metrics.get("ttft_p99_ms")),
        "itl_avg_ms": as_optional_float(bench_metrics.get("itl_avg_ms")),
        "itl_p50_ms": as_optional_float(bench_metrics.get("itl_p50_ms")),
        "itl_p99_ms": as_optional_float(bench_metrics.get("itl_p99_ms")),
        "e2e_avg_ms": as_optional_float(bench_metrics.get("e2e_avg_ms")),
        "reasons": task_reasons,
    })

overall = "PASS" if not fail_reasons else "FAIL"
summary = {
    "result": overall,
    "run_dir": str(RUN_DIR),
    "api_base": os.environ.get("API_BASE", ""),
    "model": os.environ.get("MODEL_ID", ""),
    "concurrency": as_int(os.environ.get("CONCURRENCY", 0)),
    "new_eval_url": os.environ.get("NEW_EVAL_URL", ""),
    "new_eval_sha256": os.environ.get("NEW_EVAL_SHA256", ""),
    "new_eval_git_commit": os.environ.get("NEW_EVAL_GIT_COMMIT", ""),
    "selected_tasks": [row["task"] for row in rows],
    "baseline_dir": str(BASELINE_DIR),
    "baseline_source_dir": str(BASELINE_SOURCE_DIR),
    "score_tolerance": TOL,
    "new_eval_exit_code": exit_code,
    "fail_reasons": fail_reasons,
    "speed_notes": speed_notes,
    "tasks": rows,
}

(RUN_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

with (RUN_DIR / "summary.csv").open("w", newline="") as f:
    fields = [
        "task", "status", "score", "baseline_score", "delta", "threshold",
        "sample_count", "expected_sample_count", "baseline_sample_count", "correct_count", "wrong_count",
        "failed_count", "empty_predict_count", "judge_parse_failed_count",
        "speed_metrics_available", "duration_scope", "duration_sec",
        "successful_requests", "failed_requests", "total_prompt_tokens", "total_completion_tokens",
        "request_rate_per_s", "prompt_tokens_per_s", "output_tokens_per_s", "completion_tokens_avg",
        "ttft_avg_ms", "ttft_p50_ms", "ttft_p99_ms",
        "itl_avg_ms", "itl_p50_ms", "itl_p99_ms", "e2e_avg_ms",
        "reasons",
    ]
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        out = dict(row)
        out["reasons"] = "; ".join(row["reasons"])
        writer.writerow(out)

lines = [
    f"# Result: {overall}",
    "",
    f"Run dir: `{RUN_DIR}`",
    f"API base: `{os.environ.get('API_BASE', '')}`",
    f"Model: `{os.environ.get('MODEL_ID', '')}`",
    f"Concurrency: `{os.environ.get('CONCURRENCY', '')}`",
    f"new_eval artifact: `{os.path.basename(os.environ.get('NEW_EVAL_URL', ''))}`",
    f"new_eval commit: `{os.environ.get('NEW_EVAL_GIT_COMMIT', '')}`",
    f"new_eval SHA256: `{os.environ.get('NEW_EVAL_SHA256', '')}`",
    f"Selected tasks: `{', '.join(row['task'] for row in rows)}`",
    f"Baseline dir: `{BASELINE_DIR}`",
    f"Baseline source: `{BASELINE_SOURCE_DIR}`",
    f"Score tolerance: `{TOL:.4f}` absolute",
    "",
]
if fail_reasons:
    lines += ["## Fail Reasons", ""]
    lines += [f"- {reason}" for reason in fail_reasons]
    lines += [""]

if speed_notes:
    lines += ["## Speed Notes", ""]
    lines += [f"- {note}" for note in speed_notes]
    lines += [""]

lines += [
    "## Scores",
    "",
    "| task | status | score | baseline | delta | threshold | total | expected | correct | wrong | failed | empty | judge_fail |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        f"| {row['task']} | {row['status']} | {row['score']:.4f} | "
        f"{row['baseline_score']:.4f} | {row['delta']:+.4f} | "
        f"{row['threshold']:.4f} | {row['sample_count']} | "
        f"{row['expected_sample_count']} | "
        f"{row['correct_count']} | {row['wrong_count']} | {row['failed_count']} | "
        f"{row['empty_predict_count']} | {row['judge_parse_failed_count']} |"
    )

lines += [
    "",
    "## Speed - Throughput",
    "",
    "Rollout timing only; judge time is excluded. Speed metrics are informational and do not affect accuracy PASS/FAIL.",
    "",
    "| task | scope | duration_s | success | failed | req/s | prompt tok/s | output tok/s | completion tok/req |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        f"| {row['task']} | {row['duration_scope'] or 'N/A'} | "
        f"{format_speed_float(row['duration_sec'])} | "
        f"{format_speed_int(row['successful_requests'])} | "
        f"{format_speed_int(row['failed_requests'])} | "
        f"{format_speed_float(row['request_rate_per_s'])} | "
        f"{format_speed_float(row['prompt_tokens_per_s'])} | "
        f"{format_speed_float(row['output_tokens_per_s'])} | "
        f"{format_speed_float(row['completion_tokens_avg'])} |"
    )

lines += [
    "",
    "## Speed - Latency",
    "",
    "| task | TTFT avg ms | TTFT p50 ms | TTFT p99 ms | ITL avg ms | ITL p50 ms | ITL p99 ms | E2E avg ms |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        f"| {row['task']} | "
        f"{format_speed_float(row['ttft_avg_ms'])} | "
        f"{format_speed_float(row['ttft_p50_ms'])} | "
        f"{format_speed_float(row['ttft_p99_ms'])} | "
        f"{format_speed_float(row['itl_avg_ms'])} | "
        f"{format_speed_float(row['itl_p50_ms'])} | "
        f"{format_speed_float(row['itl_p99_ms'])} | "
        f"{format_speed_float(row['e2e_avg_ms'])} |"
    )

(RUN_DIR / "summary.md").write_text("\n".join(lines) + "\n")
print((RUN_DIR / "summary.md").read_text())
PY
