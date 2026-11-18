#!/usr/bin/env bash
# Install day0, vLLM and the dataset tools on a GPU host (see docs/runbook_gpu.md, step 2).
#
# Usage:  bash scripts/gpu_setup.sh
#         VLLM_VERSION=<version> bash scripts/gpu_setup.sh     # pin another vLLM version
#
# The default vLLM version is the one this repository was checked against (docs/decisions.md,
# D-34). Its wheels need CUDA 13, so the NVIDIA driver must be 580 or newer. The script checks
# this first. DAY0_SKIP_DRIVER_CHECK=1 skips the check. The script stops at the first error and
# prints each step.
set -euo pipefail

step() { printf '\n==> %s\n' "$*"; }
die() { echo "error: $*" >&2; exit 1; }

VLLM_VERSION="${VLLM_VERSION:-0.34.0}"
MIN_DRIVER_MAJOR=580

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

step "check the NVIDIA driver (vLLM ${VLLM_VERSION} needs driver ${MIN_DRIVER_MAJOR} or newer)"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found. Run this script on the GPU host."
DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null \
  | head -n 1 | tr -d ' ' || true)"
DRIVER_MAJOR="${DRIVER_VERSION%%.*}"
[[ "$DRIVER_MAJOR" =~ ^[0-9]+$ ]] \
  || die "could not read the driver version from nvidia-smi (it printed: '${DRIVER_VERSION}')"
if (( DRIVER_MAJOR < MIN_DRIVER_MAJOR )); then
  if [[ "${DAY0_SKIP_DRIVER_CHECK:-0}" == "1" ]]; then
    echo "warning: driver ${DRIVER_VERSION} is older than ${MIN_DRIVER_MAJOR}; continuing" \
      "because DAY0_SKIP_DRIVER_CHECK=1" >&2
  else
    die "driver ${DRIVER_VERSION} is older than ${MIN_DRIVER_MAJOR}. Use a pod whose host has a" \
      "newer driver, or set DAY0_SKIP_DRIVER_CHECK=1 to try anyway."
  fi
fi
echo "driver ${DRIVER_VERSION}"

step "install uv (if missing)"
if ! command -v uv >/dev/null 2>&1; then
  # pip --user first. Some images refuse it (PEP 668), so fall back to uv's own installer.
  python3 -m pip install --user --quiet uv 2>/dev/null \
    || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

step "create the virtual environment (.venv, Python 3.11)"
if [[ -x .venv/bin/python ]]; then
  echo ".venv already exists; reusing it"
else
  uv venv --python 3.11 .venv
fi

step "install vLLM ${VLLM_VERSION}, and day0 in editable mode with the dev and suites extras"
# One resolve for all three, so the pins are checked against each other.
uv pip install -p .venv/bin/python "vllm==${VLLM_VERSION}" -e ".[dev,suites]"

step "check the GPU"
nvidia-smi

step "check the installed versions"
.venv/bin/vllm --version
.venv/bin/day0 --version
.venv/bin/python -c "import datasets; print('datasets', datasets.__version__)"

step "check that this vLLM build lists the flags day0 uses"
HELP_TEXT="$(.venv/bin/vllm serve --help=all 2>/dev/null || true)"
for flag in --logprobs-mode --max-logprobs --seed --max-model-len; do
  grep -q -- "$flag" <<<"$HELP_TEXT" \
    || die "vLLM ${VLLM_VERSION} does not list ${flag}. Choose another VLLM_VERSION."
done

echo
echo "Setup complete. Activate with: source .venv/bin/activate"
