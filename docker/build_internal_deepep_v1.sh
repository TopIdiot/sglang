#!/usr/bin/env bash
set -euo pipefail

# Invoked by the root Dockerfile's disposable DeepEP builder stage.
# Keep the private Git authentication aligned with mmq-docker for builds running
# through the internal image plugin, whose current invocation does not inject
# BuildKit secrets.
: "${VENV_PATH:?VENV_PATH must point to the image Python environment}"

GDRCOPY_VERSION="${GDRCOPY_VERSION:-2.4.4}"
GDRCOPY_SHA256="${GDRCOPY_SHA256:-8802f7bc4a589a610118023bdcdd83c10a56dea399acf6eeaac32e8cc10739a8}"
NVSHMEM_COMMIT="${NVSHMEM_COMMIT:-6b9163524793307d7b2ab914e9f7df42e0cd5fec}"
DEEPEP_V1_TAG="${DEEPEP_V1_TAG:-tag-R02C22}"
DEEPEP_BUILD_JOBS="${DEEPEP_BUILD_JOBS:-8}"
GIT_WOA_PRIVATE_TOKEN="w9AHfWKLwi252WaB5ZY6"
GIT_WOA_BASE_URL="https://private:${GIT_WOA_PRIVATE_TOKEN}@git.woa.com"

internal_no_proxy="localhost,127.0.0.1,git.woa.com,.woa.com,.tencent.com"
export no_proxy="${no_proxy:+${no_proxy},}${internal_no_proxy}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${internal_no_proxy}"

build_root="$(mktemp -d)"
trap 'rm -rf "${build_root}"' EXIT

# Build only the GDRCopy userspace library. The kernel module belongs on the
# host and must be exposed to the container as /dev/gdrdrv.
curl -fsSL \
    "https://github.com/NVIDIA/gdrcopy/archive/refs/tags/v${GDRCOPY_VERSION}.tar.gz" \
    -o "${build_root}/gdrcopy.tar.gz"
echo "${GDRCOPY_SHA256}  ${build_root}/gdrcopy.tar.gz" | sha256sum --check
tar -xzf "${build_root}/gdrcopy.tar.gz" -C "${build_root}"
gdrcopy_src="${build_root}/gdrcopy-${GDRCOPY_VERSION}"
make -C "${gdrcopy_src}" -j"${DEEPEP_BUILD_JOBS}" \
    CUDA=/usr/local/cuda prefix=/opt/gdrcopy lib_install
test -s /opt/gdrcopy/lib/libgdrapi.so.2.4
test -L /opt/gdrcopy/lib/libgdrapi.so.2
test -L /opt/gdrcopy/lib/libgdrapi.so

GIT_TERMINAL_PROMPT=0 git clone \
    "${GIT_WOA_BASE_URL}/astral/trmt/trmt-shmem.git" \
    "${build_root}/nvshmem"
git -C "${build_root}/nvshmem" checkout --detach "${NVSHMEM_COMMIT}"
rm -f "${build_root}/nvshmem/git_commit.txt" \
    "${build_root}/nvshmem"/log_*.txt

nvshmem_build_dir="${build_root}/nvshmem/build"

# Match mmq-docker's known-good NVSHMEM configuration. In particular, do not
# pass CMAKE_BUILD_TYPE: the internal packaging code preserves that spelling
# while CMake lowercases its export filenames, so "Release" breaks install.
CUDA_HOME=/usr/local/cuda \
GDRCOPY_HOME="${gdrcopy_src}" \
NVSHMEM_SHMEM_SUPPORT=0 \
NVSHMEM_UCX_SUPPORT=0 \
NVSHMEM_USE_NCCL=0 \
NVSHMEM_MPI_SUPPORT=0 \
NVSHMEM_IBGDA_SUPPORT=1 \
NVSHMEM_PMIX_SUPPORT=0 \
NVSHMEM_TIMEOUT_DEVICE_POLLING=0 \
NVSHMEM_USE_GDRCOPY=1 \
cmake -S "${build_root}/nvshmem" -B "${nvshmem_build_dir}" \
    -DCMAKE_INSTALL_PREFIX=/opt/nvshmem \
    -DNVSHMEM_BUILD_EXAMPLES=OFF \
    -DNVSHMEM_BUILD_PERFTEST=OFF \
    -DNVSHMEM_BUILD_TESTS=OFF
cmake --build "${nvshmem_build_dir}" \
    --parallel "${DEEPEP_BUILD_JOBS}"
cmake --install "${nvshmem_build_dir}"
test -s /opt/nvshmem/lib/libnvshmem.a
test -s /opt/nvshmem/lib/libnvshmem_device.a
test -e /opt/nvshmem/lib/nvshmem_bootstrap_uid.so
test -e /opt/nvshmem/lib/nvshmem_transport_ibgda.so
grep -qx "${NVSHMEM_COMMIT}" /opt/nvshmem/git_commit.txt

GIT_TERMINAL_PROMPT=0 git clone --depth 1 --branch "${DEEPEP_V1_TAG}" \
    "${GIT_WOA_BASE_URL}/astral/trmt/trmt-deepep.git" \
    "${build_root}/deepep_v1"
deepep_src="${build_root}/deepep_v1"
sed -i '/#include <cuda_bf16.h>/a#include <cuda_fp16.h>' \
    "${deepep_src}/csrc/kernels/configs.cuh"
sed -i '/case CUDA_R_16BF:/i\        case CUDA_R_16F: case_macro(__half);              \\' \
    "${deepep_src}/csrc/kernels/launch.cuh"
sed -i -e '/^        case 2560: case_macro(2560); \\$/i\' \
    -e '        case 2048: case_macro(2048); \\' \
    "${deepep_src}/csrc/kernels/launch.cuh"
grep -q '#include <cuda_fp16.h>' "${deepep_src}/csrc/kernels/configs.cuh"
grep -q 'case CUDA_R_16F' "${deepep_src}/csrc/kernels/launch.cuh"
grep -q 'case 2048' "${deepep_src}/csrc/kernels/launch.cuh"

mkdir -p /opt/deepep-v1-wheel
cd "${deepep_src}"
CUDA_HOME=/usr/local/cuda \
GDRCOPY_HOME=/opt/gdrcopy \
NVSHMEM_DIR=/opt/nvshmem \
LD_LIBRARY_PATH="/opt/nvshmem/lib:/opt/gdrcopy/lib:${LD_LIBRARY_PATH:-}" \
MAX_JOBS="${DEEPEP_BUILD_JOBS}" \
TORCH_CUDA_ARCH_LIST=9.0 \
NUM_WARPS_PER_GROUP=10 \
NUM_WARP_GROUPS=3 \
NUM_MAX_TOPK=10 \
"${VENV_PATH}/bin/python" setup.py bdist_wheel \
    --dist-dir /opt/deepep-v1-wheel
test "$(find /opt/deepep-v1-wheel -maxdepth 1 -name 'deep_ep-*.whl' | wc -l)" -eq 1
