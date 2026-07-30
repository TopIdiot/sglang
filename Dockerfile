# syntax=docker/dockerfile:1.7
FROM --platform=$BUILDPLATFORM golang:1.25.3-bookworm AS go-builder

WORKDIR /go-build

ARG TARGETOS=linux
ARG TARGETARCH=amd64

COPY 3rdparty/sr/go.mod /go-build/sr/go.mod
COPY 3rdparty/jfs-fast-get/go.mod 3rdparty/jfs-fast-get/go.sum /go-build/jfs-fast-get/
COPY 3rdparty/jfs-fast-get/third_party/juicefs/go.mod 3rdparty/jfs-fast-get/third_party/juicefs/go.sum /go-build/jfs-fast-get/third_party/juicefs/
COPY 3rdparty/jfs-fast-get/third_party/gosigar/go.mod /go-build/jfs-fast-get/third_party/gosigar/go.mod
COPY 3rdparty/remote-exec/go.mod 3rdparty/remote-exec/go.sum /go-build/remote-exec/

RUN --mount=type=cache,target=/go/pkg/mod \
    --mount=type=cache,target=/root/.cache/go-build \
    cd /go-build/sr && go mod download && \
    cd /go-build/jfs-fast-get && go mod download && \
    cd /go-build/remote-exec && go mod download

COPY 3rdparty/sr /go-build/sr
COPY 3rdparty/jfs-fast-get /go-build/jfs-fast-get
COPY 3rdparty/remote-exec /go-build/remote-exec

RUN --mount=type=cache,target=/go/pkg/mod \
    --mount=type=cache,target=/root/.cache/go-build \
    cd /go-build/sr && \
    CGO_ENABLED=0 GOOS=${TARGETOS} GOARCH=${TARGETARCH} \
        go build -trimpath -ldflags="-s -w" -o /out/sr . && \
    cd /go-build/jfs-fast-get && \
    CGO_ENABLED=0 GOOS=${TARGETOS} GOARCH=${TARGETARCH} \
        go build -trimpath -ldflags="-s -w" -o /out/jfs-fast-get ./cmd/jfs-fast-get && \
    cd /go-build/remote-exec && \
    CGO_ENABLED=0 GOOS=${TARGETOS} GOARCH=${TARGETARCH} \
        go build -trimpath -ldflags="-s -w" -o /out/remote-exec .

FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

ARG PYTHON_VERSION=3.12
ARG VENV_PATH=/envs/venv
ARG TCCL_VERSION=2.28
ARG PYTORCH_CUDA_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG SGLANG_KERNEL_INDEX_URL=https://docs.sglang.ai/whl/cu129/
ARG SGLANG_VERSION=0.0.0.dev0

ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=${VENV_PATH} \
    PATH="${VENV_PATH}/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/tccl/lib:/usr/local/cuda/lib64" \
    NCCL_IB_TC=160 \
    NCCL_SOCKET_IFNAME=bond1 \
    NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_7,mlx5_bond_8 \
    NCCL_IB_GID_INDEX=3 \
    NCCL_IB_QPS_PER_CONNECTION=4 \
    NCCL_IB_TIMEOUT=22 \
    NCCL_IB_SL=3 \
    NCCL_IB_DISABLE=0 \
    GLOO_SOCKET_IFNAME=bond1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN --mount=type=cache,target=/var/cache/apt \
    for cmd in /usr/bin/systemctl /usr/sbin/service /usr/sbin/invoke-rc.d /usr/sbin/update-rc.d; do \
        dpkg-divert --local --rename --add "${cmd}"; \
        printf '#!/bin/sh\nexit 0\n' > "${cmd}" && \
        chmod +x "${cmd}"; \
    done && \
    printf '#!/bin/sh\nexit 101\n' > /usr/sbin/policy-rc.d && \
    chmod +x /usr/sbin/policy-rc.d && \
    apt-get update && \
    apt-get install -y \
        bash \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        libibverbs-dev \
        libasio-dev \
        libboost-all-dev \
        libcurl4-openssl-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libgrpc++-dev \
        libgrpc-dev \
        libgtest-dev \
        libhiredis-dev \
        libjsoncpp-dev \
        libmsgpack-dev \
        libnuma-dev \
        libnuma1 \
        libprotobuf-dev \
        libssl-dev \
        libunwind-dev \
        liburing-dev \
        libxxhash-dev \
        libyaml-cpp-dev \
        libzstd-dev \
        ninja-build \
        patchelf \
        pkg-config \
        protobuf-compiler \
        protobuf-compiler-grpc \
        python3 \
        python3-dev \
        unzip \
        wget \
        xz-utils \
        zstd \
        bzip2 \
        dstat \
        ffmpeg \
        htop && \
    cd /var/lib/pcp/pmns && /usr/lib/pcp/pmns/Rebuild && \
    rm -f /usr/sbin/policy-rc.d && \
    for cmd in /usr/bin/systemctl /usr/sbin/service /usr/sbin/invoke-rc.d /usr/sbin/update-rc.d; do \
        rm -f "${cmd}"; \
        if dpkg-divert --list "${cmd}" >/dev/null; then \
            dpkg-divert --rename --remove "${cmd}"; \
        fi; \
    done && \
    rm -rf /var/lib/apt/lists/*

COPY --from=go-builder /out/sr /usr/local/bin/sr
COPY --from=go-builder /out/jfs-fast-get /usr/local/bin/jfs-fast-get
COPY --from=go-builder /out/remote-exec /usr/local/bin/remote-exec

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --no-modify-path --profile minimal
ENV PATH="/root/.local/bin:/root/.cargo/bin:${PATH}"

# Inline create-venv.sh's venv creation and managed-Python embedding so this
# Dockerfile does not depend on a prebuilt venv.tar or an external script.
RUN --mount=type=cache,target=/root/.cache/uv <<'BASH'
set -euo pipefail
curl -LsSf https://astral.sh/uv/install.sh | sh
/root/.local/bin/uv venv \
        --relocatable \
        --seed \
        --clear \
        --managed-python \
        --python "${PYTHON_VERSION}" \
        "${VENV_PATH}"
venv_python="${VENV_PATH}/bin/python"
base_prefix="$("${venv_python}" -c 'import sys; print(sys.base_prefix)')"
python_exe_name="$("${venv_python}" -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')"
rm -rf "${VENV_PATH}/.python"
mkdir -p "${VENV_PATH}/.python"
cp -a "${base_prefix}/." "${VENV_PATH}/.python/"
"${venv_python}" - <<'PY' "${VENV_PATH}/.python"
import pathlib
import sys

embedded_prefix = pathlib.Path(sys.argv[1]).resolve()
for path in (embedded_prefix / "lib").glob("python*/_sysconfigdata*.py"):
    text = path.read_text()
    marker = "build_time_vars = {"
    if "_sglang_embedded_prefix" in text or marker not in text:
        continue
    text = text.replace(
        marker,
        "import pathlib as _sglang_pathlib\n"
        "_sglang_embedded_prefix = _sglang_pathlib.Path(__file__).resolve().parents[2]\n"
        + marker,
        1,
    )
    text += (
        "\n_sglang_include = str(_sglang_embedded_prefix / 'include' / "
        "('python' + build_time_vars.get('LDVERSION', build_time_vars.get('VERSION', ''))))\n"
        "for _sglang_key in ('INCLUDEPY', 'CONFINCLUDEPY'):\n"
        "    if _sglang_key in build_time_vars:\n"
        "        build_time_vars[_sglang_key] = _sglang_include\n"
    )
    path.write_text(text)
PY
ln -sfn "../.python/bin/${python_exe_name}" "${VENV_PATH}/bin/python"
ln -sfn python "${VENV_PATH}/bin/python3"
ln -sfn python "${VENV_PATH}/bin/${python_exe_name}"
BASH

WORKDIR /sgl-workspace

COPY python/pyproject.toml /sgl-workspace/python/pyproject.toml

RUN "${VENV_PATH}/bin/python" - <<'PY'
import pathlib
import re

path = pathlib.Path("/sgl-workspace/python/pyproject.toml")
text = path.read_text()

def replace_dependency_line(text: str, package: str, replacement: str) -> str:
    pattern = re.compile(
        rf'(?m)^(\s*)"{re.escape(package)}(?:\[[^\]]+\])?[^"]*",(\s*(?:#.*)?)$'
    )
    text, count = pattern.subn(rf'\1"{replacement}",\2', text, count=1)
    if count != 1:
        raise SystemExit(f"expected dependency for {package!r} not found in {path}")
    return text

for package, replacement in (
    ("cuda-python", "cuda-python>=12,<13"),
    ("sglang-kernel", "sglang-kernel==0.4.2.post2+cu129"),
    ("nvidia-cutlass-dsl", "nvidia-cutlass-dsl==4.5.0"),
):
    text = replace_dependency_line(text, package, replacement)
path.write_text(text)
PY

# Install torch first; this is the largest and most stable dependency layer.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python "${VENV_PATH}/bin/python" \
        --index-url "${PYTORCH_CUDA_INDEX_URL}" \
        "torch==2.11.0" \
        "torchvision==0.26.0" \
        "torchaudio==2.11.0"

# TCCL must be installed after torch is importable, before the rest of Python
# dependencies pull in or compile against communication libraries.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python "${VENV_PATH}/bin/python" requests

RUN mkdir -p /src/tccl && cd /src/tccl && \
    wget -O tccl_install.py \
        https://mirrors.tencent.com/repository/generic/TCCL/NV/tccl_install.py && \
    "${VENV_PATH}/bin/python" tccl_install.py -v "${TCCL_VERSION}" --install_mode compile_install --skip_check --clean 2>&1 | tee /tmp/tccl_install.log && \
    ! grep -q "TCCL Install.*ERROR" /tmp/tccl_install.log && \
    grep -q "Successfully installed" /tmp/tccl_install.log && \
    test -s /usr/local/tccl/lib/libtccl.so.2 && \
    test -s /usr/local/tccl/lib/plugin/libnccl-profiler-inspector.so && \
    test -s /usr/local/tccl/lib/plugin/libnccl-tuner-astralNet.so && \
    cd / && rm -rf /src/tccl

COPY 3rdparty/Mooncake /sgl-workspace/3rdparty/Mooncake

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/uv \
    cd /sgl-workspace/3rdparty/Mooncake && \
    if [ ! -f extern/pybind11/CMakeLists.txt ]; then \
        rm -rf extern/pybind11 && \
        git clone --depth=1 --branch stable https://github.com/pybind/pybind11.git extern/pybind11; \
    fi && \
    if [ ! -f extern/yalantinglibs/CMakeLists.txt ]; then \
        rm -rf extern/yalantinglibs && \
        git clone --depth=1 --branch v0.5.7 https://github.com/alibaba/yalantinglibs.git extern/yalantinglibs; \
    fi && \
    cmake -S extern/yalantinglibs -B extern/yalantinglibs/build \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_BENCHMARK=OFF \
        -DBUILD_UNIT_TESTS=OFF && \
    cmake --build extern/yalantinglibs/build -j"$(nproc)" && \
    cmake --install extern/yalantinglibs/build && \
    if [ -f /usr/local/cuda/lib64/stubs/libcuda.so ] && [ ! -e /usr/local/cuda/lib64/stubs/libcuda.so.1 ]; then \
        ln -s libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1; \
    fi && \
    export LIBRARY_PATH="/usr/local/cuda/lib64/stubs:${LIBRARY_PATH:-}" && \
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}" && \
    cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
        -DPython3_EXECUTABLE="${VENV_PATH}/bin/python" \
        -DBUILD_BENCHMARK=OFF \
        -DBUILD_UNIT_TESTS=OFF \
        -DUSE_CUDA=ON \
        -DWITH_EP=OFF \
        -DWITH_STORE_RUST=OFF && \
    cmake --build build -j"$(nproc)" && \
    PYTHON_VERSION="${PYTHON_VERSION}" ./scripts/build_wheel.sh "${PYTHON_VERSION}" dist && \
    uv pip install --python "${VENV_PATH}/bin/python" mooncake-wheel/dist/*.whl && \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}" "${VENV_PATH}/bin/python" - <<'PY' && \
    rm -rf build extern/yalantinglibs/build mooncake-wheel/build mooncake-wheel/dist mooncake-wheel/repaired_wheels_*
from importlib.metadata import version

import mooncake
import mooncake.engine

print(f"mooncake-transfer-engine={version('mooncake-transfer-engine')}")
print(f"mooncake_module={mooncake.__file__}")
print(f"mooncake_engine={mooncake.engine.__file__}")
PY

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python "${VENV_PATH}/bin/python" \
        --index-url "${SGLANG_KERNEL_INDEX_URL}" \
        "sglang-kernel==0.4.2.post2+cu129"

RUN "${VENV_PATH}/bin/python" - <<'PY' > /tmp/sglang-runtime-requirements.txt
import pathlib
import tomllib

pyproject = pathlib.Path("/sgl-workspace/python/pyproject.toml")
data = tomllib.loads(pyproject.read_text())
deep_gemm_version = None
for dep in data["project"]["dependencies"]:
    dep_lower = dep.strip().lower()
    if dep_lower.startswith("cuda-python"):
        print("cuda-python>=12,<13")
    elif dep_lower == "numpy":
        print("numpy==2.2.6")
    elif dep_lower.startswith(("torch==", "torchaudio==", "torchvision")):
        continue
    elif dep_lower.startswith("sglang-kernel=="):
        continue
    elif dep_lower.startswith("sgl-deep-gemm=="):
        deep_gemm_version = dep.split("==", 1)[1].strip()
        continue
    else:
        print(dep)
if not deep_gemm_version:
    raise SystemExit("expected sgl-deep-gemm dependency not found")
pathlib.Path("/tmp/sgl-deep-gemm-version.txt").write_text(deep_gemm_version)
PY

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python "${VENV_PATH}/bin/python" \
        -r /tmp/sglang-runtime-requirements.txt \
        decord2

RUN --mount=type=cache,target=/root/.cache/uv <<'BASH'
set -euo pipefail
SGL_DEEP_GEMM_VERSION="$(cat /tmp/sgl-deep-gemm-version.txt)"
uv pip install --python "${VENV_PATH}/bin/python" \
    "https://github.com/sgl-project/whl/releases/download/v${SGL_DEEP_GEMM_VERSION}/sgl_deep_gemm-${SGL_DEEP_GEMM_VERSION}+cu129-py3-none-manylinux2014_$(uname -m).whl" \
    --force-reinstall
BASH

# DeepEP v1.2.1 is prebuilt for CPython 3.12, x86_64, CUDA 12, and SM90.
# deep_ep declares no dependencies, so its NVSHMEM runtime must be installed
# explicitly. The assertion below verifies that the wheel retains NVSHMEM
# linkage required by the internode/low-latency kernels.
RUN --mount=type=bind,source=3rdparty/deep_ep_wheel/deep_ep-1.2.1-cp312-cp312-linux_x86_64.whl,target=/tmp/deep_ep-1.2.1-cp312-cp312-linux_x86_64.whl \
    --mount=type=cache,target=/root/.cache/uv <<'BASH'
set -euo pipefail
DEEP_EP_WHEEL=/tmp/deep_ep-1.2.1-cp312-cp312-linux_x86_64.whl
echo "28e6f24c2d9de18d3eb0640421d96ff8d1f8744830ee5ed83d3af0a6fdaf8bcb  ${DEEP_EP_WHEEL}" | sha256sum --check
uv pip install --python "${VENV_PATH}/bin/python" \
    nvidia-nvshmem-cu12==3.4.5 \
    "${DEEP_EP_WHEEL}"
"${VENV_PATH}/bin/python" - <<'PY'
import pathlib
import subprocess

import deep_ep
from deep_ep import Buffer  # noqa: F401

site = pathlib.Path(deep_ep.__file__).resolve().parent.parent
so = next(site.glob("deep_ep_cpp*.so"))
linked = subprocess.run(["ldd", str(so)], capture_output=True, text=True).stdout
if "libnvshmem_host" not in linked:
    raise SystemExit(
        "deep_ep was built WITHOUT NVSHMEM (internode/low-latency disabled)"
    )
print(f"deep_ep={deep_ep.__version__ if hasattr(deep_ep, '__version__') else '1.2.1'}, nvshmem linkage OK")
PY
BASH

# MK compiles CUDA kernels at runtime. Keep its source tree in the image so the
# editable package can resolve the JIT headers under csrcs/include.
COPY 3rdparty/mk/pyproject.toml 3rdparty/mk/README.md /sgl-workspace/3rdparty/mk/
COPY 3rdparty/mk/mk /sgl-workspace/3rdparty/mk/mk
COPY 3rdparty/mk/csrcs /sgl-workspace/3rdparty/mk/csrcs

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python "${VENV_PATH}/bin/python" \
        -e /sgl-workspace/3rdparty/mk && \
    "${VENV_PATH}/bin/python" - <<'PY'
import mk
from mk.jit.compiler import REPO_ROOT

jit_header = REPO_ROOT / "csrcs/include/mk/context_impl.cuh"
if not jit_header.is_file():
    raise SystemExit(f"MK JIT header not found: {jit_header}")
print(f"mk_module={mk.__file__}")
print(f"mk_jit_header={jit_header}")
PY

COPY sgl-model-gateway /sgl-workspace/sgl-model-gateway

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/root/.cargo/git \
    --mount=type=cache,target=/sgl-workspace/sgl-model-gateway/target \
    uv pip install --python "${VENV_PATH}/bin/python" "maturin<1.14" && \
    cd /sgl-workspace/sgl-model-gateway/bindings/python && \
    ulimit -n 65536 && \
    "${VENV_PATH}/bin/python" -m maturin build \
        --release \
        --features vendored-openssl \
        --out dist && \
    uv pip install --python "${VENV_PATH}/bin/python" dist/*.whl && \
    "${VENV_PATH}/bin/python" - <<'PY'
from importlib.metadata import PackageNotFoundError, version

import sglang_router
import sglang_router.launch_router
import sglang_router.sglang_router_rs

for router_dist_name in ("welm-sglang-router", "sglang-router"):
    try:
        router_version = version(router_dist_name)
        break
    except PackageNotFoundError:
        pass
else:
    raise SystemExit("Neither welm-sglang-router nor sglang-router is installed")
print(f"{router_dist_name}={router_version}")
print(f"sglang_router_module={sglang_router.__file__}")
PY

COPY README.md LICENSE /sgl-workspace/
COPY python /sgl-workspace/python
COPY proto /sgl-workspace/proto
COPY rust/sglang-grpc /sgl-workspace/rust/sglang-grpc

RUN "${VENV_PATH}/bin/python" - <<'PY'
import pathlib
import re

path = pathlib.Path("/sgl-workspace/python/pyproject.toml")
text = path.read_text()

def replace_dependency_line(text: str, package: str, replacement: str) -> str:
    pattern = re.compile(
        rf'(?m)^(\s*)"{re.escape(package)}(?:\[[^\]]+\])?[^"]*",(\s*(?:#.*)?)$'
    )
    text, count = pattern.subn(rf'\1"{replacement}",\2', text, count=1)
    if count != 1:
        raise SystemExit(f"expected dependency for {package!r} not found in {path}")
    return text

for package, replacement in (
    ("cuda-python", "cuda-python>=12,<13"),
    ("sglang-kernel", "sglang-kernel==0.4.2.post2+cu129"),
    ("nvidia-cutlass-dsl", "nvidia-cutlass-dsl==4.5.0"),
):
    text = replace_dependency_line(text, package, replacement)
path.write_text(text)
PY

RUN --mount=type=cache,target=/root/.cache/uv \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG="${SGLANG_VERSION}" \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_WELM_SGLANG="${SGLANG_VERSION}" \
    uv pip install --python "${VENV_PATH}/bin/python" --no-deps /sgl-workspace/python && \
    uv pip check --python "${VENV_PATH}/bin/python" && \
    "${VENV_PATH}/bin/python" - <<'PY'
from importlib.metadata import PackageNotFoundError, version
import pathlib
import sys
import sysconfig

import deep_ep
import deep_gemm
import decord
import sglang
import torch

include_dir = pathlib.Path(sysconfig.get_config_var("INCLUDEPY"))
python_h = include_dir / "Python.h"
if not python_h.is_file():
    raise SystemExit(f"Python.h not found at {python_h}")
cuda_python = version("cuda-python")
if not cuda_python.startswith("12."):
    raise SystemExit(f"cuda-python must be 12.x, got {cuda_python}")
if not str(torch.version.cuda).startswith("12."):
    raise SystemExit(f"torch CUDA runtime must be 12.x, got {torch.version.cuda}")
for sglang_dist_name in ("welm-sglang", "sglang"):
    try:
        sglang_version = version(sglang_dist_name)
        break
    except PackageNotFoundError:
        pass
else:
    raise SystemExit("Neither welm-sglang nor sglang is installed")
print(f"python={sys.executable}")
print(f"python_include={include_dir}")
print(f"{sglang_dist_name}={sglang_version}")
print(f"torch={torch.__version__}, torch_cuda={torch.version.cuda}")
print(f"cuda-python={cuda_python}")
print(f"decord_module={decord.__file__}")
print(f"deep_gemm={version('sgl-deep-gemm')}, deep_gemm_module={deep_gemm.__file__}")
print(f"deep_ep={version('deep-ep')}, nvshmem={version('nvidia-nvshmem-cu12')}")
print(f"sglang_module={sglang.__file__}")
PY

WORKDIR /sgl-workspace
CMD ["/bin/bash"]
