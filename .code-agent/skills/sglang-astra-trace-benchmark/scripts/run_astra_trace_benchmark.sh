#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_astra_trace_benchmark.sh --base-url URL [options]

Required:
  --base-url URL       SGLang server URL, /v1 base URL, or chat completions URL

Options:
  --concurrency N      Number of conversation-serial clients (default: 8)
  --sample-num N       Limit the run to N measured requests (default: all)
  --trace PATH         Local Astra JSONL/JSONL.GZ override
  --model MODEL        Served model id (default: first id returned by /v1/models)
  --run-root DIR       Parent directory for timestamped runs
  --prepare-only       Generate the run command without downloading or running
  -h, --help           Show this help

Authentication:
  Set SGLANG_API_KEY; its value is never written to command.txt or run.log.
EOF
}

die() {
  printf 'astra trace benchmark: %s\n' "$*" >&2
  exit 2
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
skill_dir="$(cd -- "${script_dir}/.." && pwd -P)"

artifact_url="https://mirrors.tencent.com/repository/generic/welm/astra_trace_bench/bin/20260730/astra-trace-bench-linux-amd64-690b8fa6"
artifact_sha256="690b8fa60a60663e0f40a2d59af8f81ec54897c5073730423d51a2b2778aed29"
source_git_commit="7dc430d324770da606a81ce042a0a15fad7dbee4"
embedded_trace_file="welm_2026-07-12_10-00_1000_no_decode1.jsonl"
trace_sha256="2e93cac34d6e8d2ae0f89769465cb6bf9048fd2d6dd6c332123cf2adf693106a"

base_url=""
concurrency=8
sample_num=""
trace_path=""
model=""
run_root="${ASTRA_TRACE_BENCH_RUN_ROOT:-/tmp/astra_trace_bench}"
prepare_only=0

while (($# > 0)); do
  case "$1" in
    --base-url)
      (($# >= 2)) || die "--base-url requires a value"
      base_url="$2"
      shift 2
      ;;
    --concurrency)
      (($# >= 2)) || die "--concurrency requires a value"
      concurrency="$2"
      shift 2
      ;;
    --sample-num)
      (($# >= 2)) || die "--sample-num requires a value"
      sample_num="$2"
      shift 2
      ;;
    --trace)
      (($# >= 2)) || die "--trace requires a value"
      trace_path="$2"
      shift 2
      ;;
    --model)
      (($# >= 2)) || die "--model requires a value"
      model="$2"
      shift 2
      ;;
    --run-root)
      (($# >= 2)) || die "--run-root requires a value"
      run_root="$2"
      shift 2
      ;;
    --prepare-only)
      prepare_only=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

[[ -n "${base_url}" ]] || die "--base-url is required"
[[ "${concurrency}" =~ ^[1-9][0-9]*$ ]] || die "--concurrency must be a positive integer"
if [[ -n "${sample_num}" ]]; then
  [[ "${sample_num}" =~ ^[1-9][0-9]*$ ]] || die "--sample-num must be a positive integer"
fi
if [[ -n "${trace_path}" && ! -f "${trace_path}" ]]; then
  die "trace not found: ${trace_path}"
fi
command -v curl >/dev/null 2>&1 || die "curl is required"
command -v python3 >/dev/null 2>&1 || die "python3 is required"
command -v sha256sum >/dev/null 2>&1 || die "sha256sum is required"

base_url="${base_url%/}"
case "${base_url}" in
  */v1/chat/completions)
    endpoint="${base_url}"
    api_base="${base_url%/chat/completions}"
    ;;
  */v1)
    api_base="${base_url}"
    endpoint="${base_url}/chat/completions"
    ;;
  *)
    api_base="${base_url}/v1"
    endpoint="${base_url}/v1/chat/completions"
    ;;
esac

run_id="$(date -u '+%Y%m%d_%H%M%S')_pid$$"
run_dir="${run_root%/}/${run_id}"
binary_dir="${run_dir}/binary"
binary="${binary_dir}/astra-trace-bench"
input_dir="${run_dir}/input"
mkdir -p "${binary_dir}" "${input_dir}"

printf '%s\n' "${artifact_url}" >"${binary_dir}/astra-trace-bench.url.txt"
printf '%s  %s\n' "${artifact_sha256}" "astra-trace-bench" >"${binary_dir}/astra-trace-bench.sha256"
printf '%s\n' "${source_git_commit}" >"${binary_dir}/astra-trace-bench.git-commit.txt"
if [[ -z "${trace_path}" ]]; then
  printf 'embedded:%s\n' "${embedded_trace_file}" >"${input_dir}/trace.source.txt"
  printf '%s  %s\n' "${trace_sha256}" "${embedded_trace_file}" >"${input_dir}/trace.sha256"
else
  printf '%s\n' "${trace_path}" >"${input_dir}/trace.source.txt"
fi

if [[ -z "${model}" ]] && ((prepare_only == 0)); then
  model_curl=(curl --noproxy '*' -fsS)
  if [[ -n "${SGLANG_API_KEY:-}" ]]; then
    model_curl+=(-H "Authorization: Bearer ${SGLANG_API_KEY}")
  fi
  if "${model_curl[@]}" "${api_base}/models" >"${run_dir}/models.json"; then
    model="$(python3 -c 'import json,sys; d=json.load(sys.stdin); print((d.get("data") or [{}])[0].get("id", ""))' <"${run_dir}/models.json" || true)"
  else
    printf '{}\n' >"${run_dir}/models.json"
  fi
elif [[ -n "${model}" ]]; then
  printf '{}\n' >"${run_dir}/models.json"
else
  printf '{}\n' >"${run_dir}/models.json"
fi
model="${model:-welmv4}"

if [[ -z "${trace_path}" ]]; then
  summary_trace_source="embedded:${artifact_url}#${embedded_trace_file}"
  summary_trace_sha256="${trace_sha256}"
else
  summary_trace_source="custom:${trace_path}"
  summary_trace_sha256="$(sha256sum "${trace_path}" | awk '{print $1}')"
fi

cmd=(
  "${binary}"
  --endpoint "${endpoint}"
  --model "${model}"
  --concurrent "${concurrency}"
)
if [[ -n "${trace_path}" ]]; then
  cmd+=(--input "${trace_path}")
fi
if [[ -n "${sample_num}" ]]; then
  cmd+=(--sample-num "${sample_num}")
fi
cmd+=(--output-dir "${run_dir}")

{
  if [[ -n "${SGLANG_API_KEY:-}" ]]; then
    printf 'SGLANG_API_KEY=<redacted> '
  fi
  printf '%q ' "${cmd[@]}"
  printf '\n'
} >"${run_dir}/command.txt"

printf 'run_dir=%s\n' "${run_dir}"
printf 'endpoint=%s\n' "${endpoint}"
printf 'model=%s concurrency=%s sample_num=%s\n' "${model}" "${concurrency}" "${sample_num:-all}"
printf 'trace=%s\n' "${summary_trace_source}"
printf 'artifact=%s\n' "${artifact_url}"
printf 'artifact_sha256=%s\n' "${artifact_sha256}"
printf 'source_git_commit=%s\n' "${source_git_commit}"
if [[ -z "${trace_path}" ]]; then
  printf 'embedded_trace_file=%s\n' "${embedded_trace_file}"
  printf 'trace_sha256=%s\n' "${trace_sha256}"
fi

if ((prepare_only)); then
  printf 'prepare-only: command written to %s\n' "${run_dir}/command.txt"
  exit 0
fi

download_verified() {
  local url="$1"
  local expected_sha256="$2"
  local destination="$3"
  local label="$4"
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    -u ALL_PROXY -u all_proxy \
    curl -sS --fail-with-body -L -o "${destination}.part" "${url}"
  local downloaded_sha256
  downloaded_sha256="$(sha256sum "${destination}.part" | awk '{print $1}')"
  if [[ "${downloaded_sha256}" != "${expected_sha256}" ]]; then
    rm -f "${destination}.part"
    die "${label} SHA256 mismatch: got ${downloaded_sha256}, expected ${expected_sha256}"
  fi
  mv "${destination}.part" "${destination}"
}

download_verified "${artifact_url}" "${artifact_sha256}" "${binary}" "binary artifact"
chmod +x "${binary}"
"${binary}" --help >"${binary_dir}/astra-trace-bench.help.txt" 2>&1

# Contact internal SGLang endpoints directly even when the host has proxy
# variables configured.
export NO_PROXY="*"
export no_proxy="*"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

set +e
"${cmd[@]}" 2>&1 | tee "${run_dir}/run.log"
bench_exit=${PIPESTATUS[0]}
set -e
printf '%s\n' "${bench_exit}" >"${run_dir}/bench.exitcode"

export ASTRA_ARTIFACT_URL="${artifact_url}"
export ASTRA_ARTIFACT_SHA256="${artifact_sha256}"
export ASTRA_SOURCE_GIT_COMMIT="${source_git_commit}"
export ASTRA_TRACE_SOURCE="${summary_trace_source}"
export ASTRA_TRACE_SHA256="${summary_trace_sha256}"
set +e
python3 - "${run_dir}" "${sample_num}" "${bench_exit}" "${model}" "${concurrency}" <<'PY'
import json
import os
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
requested = int(sys.argv[2]) if sys.argv[2] else None
bench_exit = int(sys.argv[3])
model = sys.argv[4]
concurrency = int(sys.argv[5])
artifact_url = os.environ["ASTRA_ARTIFACT_URL"]
artifact_sha256 = os.environ["ASTRA_ARTIFACT_SHA256"]
source_git_commit = os.environ["ASTRA_SOURCE_GIT_COMMIT"]
trace_source = os.environ["ASTRA_TRACE_SOURCE"]
trace_sha256 = os.environ["ASTRA_TRACE_SHA256"]
metrics_path = run_dir / "bench_metrics.json"
requests_path = run_dir / "requests.jsonl"
reasons = []
metrics = {}

if bench_exit != 0:
    reasons.append(f"benchmark exit code is {bench_exit}")
if not metrics_path.is_file():
    reasons.append("bench_metrics.json is missing")
else:
    try:
        document = json.loads(metrics_path.read_text())
        metrics = document.get("bench_metrics", {})
    except Exception as exc:
        reasons.append(f"cannot parse bench_metrics.json: {exc}")

sample_count = int(metrics.get("sample_count", -1))
successful = int(metrics.get("successful_requests", -1))
failed = int(metrics.get("failed_requests", -1))
empty = int(metrics.get("empty_predict_count", -1))
sample_mode = "all" if requested is None else "limit"
expected = requested
if requested is None:
    if sample_count <= 0:
        reasons.append(f"sample_count={sample_count}, expected a positive full-trace count")
elif sample_count != requested:
    reasons.append(f"sample_count={sample_count}, expected {requested}")
if successful != sample_count:
    reasons.append(f"successful_requests={successful}, expected sample_count={sample_count}")
if failed != 0:
    reasons.append(f"failed_requests={failed}, expected 0")
if empty != 0:
    reasons.append(f"empty_predict_count={empty}, expected 0")

request_lines = -1
if requests_path.is_file():
    with requests_path.open() as handle:
        request_lines = sum(1 for line in handle if line.strip())
else:
    reasons.append("requests.jsonl is missing")
if request_lines != sample_count:
    reasons.append(f"requests.jsonl has {request_lines} records, expected sample_count={sample_count}")

status = "PASS" if not reasons else "FAIL"
summary = {
    "status": status,
    "reasons": reasons,
    "run_dir": str(run_dir),
    "artifact_url": artifact_url,
    "artifact_sha256": artifact_sha256,
    "source_git_commit": source_git_commit,
    "trace_source": trace_source,
    "trace_sha256": trace_sha256,
    "model": model,
    "concurrency": concurrency,
    "sample_mode": sample_mode,
    "requested_sample_count": requested,
    "expected_sample_count": expected,
    "request_records": request_lines,
    "metrics": metrics,
}
(run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

def value(name):
    return metrics.get(name, "n/a")

lines = [
    f"# {status}: SGLang Astra trace benchmark",
    "",
    f"Run directory: `{run_dir}`",
    f"Artifact: `{artifact_url}`",
    f"Artifact SHA256: `{artifact_sha256}`",
    f"Source Git commit: `{source_git_commit}`",
    f"Trace source: `{trace_source}`",
    f"Trace SHA256: `{trace_sha256}`",
    "",
    "| Field | Value |",
    "|---|---:|",
    f"| Model | {model} |",
    f"| Configured concurrency | {concurrency} |",
    f"| Observed max concurrency | {value('concurrency_observed_max')} |",
    f"| Sample mode | {sample_mode} |",
    f"| Sample count | {value('sample_count')} |",
    f"| Successful requests | {value('successful_requests')} |",
    f"| Failed requests | {value('failed_requests')} |",
    f"| Request rate (req/s) | {value('request_rate_per_s')} |",
    f"| Output throughput (token/s) | {value('output_tokens_per_s')} |",
    f"| TTFT avg / p99 (ms) | {value('ttft_avg_ms')} / {value('ttft_p99_ms')} |",
    f"| ITL avg / p99 (ms) | {value('itl_avg_ms')} / {value('itl_p99_ms')} |",
    f"| E2E avg / p99 (ms) | {value('e2e_avg_ms')} / {value('e2e_p99_ms')} |",
]
if reasons:
    lines.extend(["", "Failure reasons:"])
    lines.extend(f"- {reason}" for reason in reasons)
(run_dir / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
raise SystemExit(0 if status == "PASS" else 1)
PY
summary_exit=$?
set -e

if ((bench_exit != 0 || summary_exit != 0)); then
  exit 1
fi
