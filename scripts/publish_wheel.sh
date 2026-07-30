#!/usr/bin/env bash
# Upload ./_wheels/welm_*.whl to tencent_pypi via twine.
#
# Env inputs:
#   TWINE_USERNAME  required
#   TWINE_PASSWORD  required
#   TWINE_REPO_URL  default: mirrors.tencent.com tencent_pypi
#   WHEELS_DIR      default: <repo>/_wheels
#   PYTHON          default: python3
set -euo pipefail
set -x
shopt -s nullglob

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

TWINE_REPO_URL="${TWINE_REPO_URL:-https://mirrors.tencent.com/repository/pypi/tencent_pypi/simple}"
WHEELS_DIR="${WHEELS_DIR:-${REPO_ROOT}/_wheels}"
PYTHON="${PYTHON:-python3}"

: "${TWINE_USERNAME:?TWINE_USERNAME not set}"
: "${TWINE_PASSWORD:?TWINE_PASSWORD not set}"

files=("${WHEELS_DIR}"/welm_*.whl)
[ ${#files[@]} -gt 0 ] || { echo "ERROR: no welm_*.whl in ${WHEELS_DIR}/" >&2; exit 1; }

"$PYTHON" -m pip install --upgrade --quiet \
    "twine>=6.1" "pkginfo>=1.11" "packaging>=24.2"

echo "=== uploading ${#files[@]} wheels as ${TWINE_USERNAME} ==="
printf '  %s\n' "${files[@]}"

"$PYTHON" -m twine upload \
    --repository-url "${TWINE_REPO_URL}" \
    --non-interactive \
    "${files[@]}"

echo "✓ ${#files[@]} wheels uploaded"
