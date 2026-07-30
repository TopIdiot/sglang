#!/usr/bin/env bash
# Build welm-sglang + welm-sglang-router wheels. welm-sglang KEEPS its runtime
# deps (single source of truth); only the CUDA-specific ones are replaced with
# the cu12 values proven by the docker_v2 training image:
#   cuda-python -> >=12,<13 ; sglang-kernel -> ==0.4.2.post1+cu129 (the CUDA 12.9 build on
#   the sgl-cu129 index) ; nvidia-cutlass-dsl -> ==4.5.0 ; transformers -> ==5.12.1
# optional-dependencies (diffusion/tracing/http2) are kept so the app can request them.
# Runs inside the wheel-builder image (see wxg-odyssey/wheel-builder repo).
#
# Env inputs:
#   SETUPTOOLS_SCM_PRETEND_VERSION_FOR_WELM_SGLANG  required
#   WHEELS_DIR                                       default: <repo>/_wheels
#   PYTHON                                           default: python3
#   MANYLINUX_PLAT                                   default: manylinux_2_28_x86_64
set -xeuo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

: "${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_WELM_SGLANG:?SETUPTOOLS_SCM_PRETEND_VERSION_FOR_WELM_SGLANG not set}"

WHEELS_DIR="${WHEELS_DIR:-${REPO_ROOT}/_wheels}"
PYTHON="${PYTHON:-python3}"
MANYLINUX_PLAT="${MANYLINUX_PLAT:-manylinux_2_28_x86_64}"
MATURIN_COMPAT="${MATURIN_COMPAT:-manylinux_2_28}"

rm -rf "${WHEELS_DIR}"
mkdir -p "${WHEELS_DIR}"

# ---------------------------------------------------------------------------
# Build-time patches:
#   python/pyproject.toml         replace cu-specific deps with cu12 values
#                                 (deps + optional-dependencies otherwise kept)
#   rust/sglang-grpc/Cargo.toml   pyo3 features += "abi3-py310"
# ---------------------------------------------------------------------------
"$PYTHON" - <<'PY'
import pathlib
import subprocess
import sys

try:
    import tomlkit
except ImportError:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "tomlkit"],
        check=True,
    )
    import tomlkit

# --- python/pyproject.toml -----------------------------------------------
# Keep deps; only replace the CUDA-specific ones with the cu12 values proven by
# the docker_v2 training image. optional-dependencies are left intact.
import re

REPLACE = {
    "cuda-python": "cuda-python>=12,<13",
    "sglang-kernel": "sglang-kernel==0.4.2.post1+cu129",
    "nvidia-cutlass-dsl": "nvidia-cutlass-dsl==4.5.0",
    "transformers": "transformers==5.12.1",
}

def dist_name(dep):
    # distribution name = text before any version/extra/marker/space
    return re.split(r"[<>=!~;\[ ]", str(dep).strip(), 1)[0].strip().lower().replace("_", "-")

py_toml = pathlib.Path("python/pyproject.toml")
doc = tomlkit.parse(py_toml.read_text())

deps = doc["project"]["dependencies"]
seen = set()
for i, dep in enumerate(deps):
    nm = dist_name(dep)
    if nm in REPLACE:
        deps[i] = REPLACE[nm]
        seen.add(nm)

missing = set(REPLACE) - seen
assert not missing, f"expected deps to replace not found in pyproject: {sorted(missing)}"

py_toml.write_text(tomlkit.dumps(doc))
print(f"=== replaced cu12 deps: {sorted(seen)} ===")

# --- rust/sglang-grpc/Cargo.toml -----------------------------------------
cargo_path = pathlib.Path("rust/sglang-grpc/Cargo.toml")
cargo = tomlkit.parse(cargo_path.read_text())
pyo3 = cargo["dependencies"]["pyo3"]
features = pyo3.get("features", tomlkit.array())
if not any(str(f).startswith("abi3-py") for f in features):
    features.append("abi3-py310")
    pyo3["features"] = features
    cargo_path.write_text(tomlkit.dumps(cargo))
PY

# ---------------------------------------------------------------------------
# welm-sglang-router (maturin)
# ---------------------------------------------------------------------------
(
    cd sgl-model-gateway/bindings/python
    ulimit -n 65536
    "$PYTHON" -m maturin build \
        --release \
        --features vendored-openssl \
        --compatibility "${MATURIN_COMPAT}" \
        --out "${WHEELS_DIR}"
)

# ---------------------------------------------------------------------------
# welm-sglang (setuptools-rust)
# ---------------------------------------------------------------------------
STAGING="${WHEELS_DIR}/_staging"
mkdir -p "${STAGING}"
(
    cd python
    "$PYTHON" -m build --wheel --outdir "${STAGING}" \
        --config-setting=--build-option=--py-limited-api=cp310
)

for whl in "${STAGING}"/welm_sglang-*.whl; do
    "$PYTHON" -m auditwheel show "$whl" || true
    "$PYTHON" -m auditwheel repair \
        --plat "${MANYLINUX_PLAT}" \
        --strip \
        -w "${WHEELS_DIR}" \
        "$whl"
done
rm -rf "${STAGING}"

echo "=== Built wheels ==="
ls -la "${WHEELS_DIR}/"
