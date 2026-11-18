#!/usr/bin/env bash
# Install day0 and a pinned vLLM on a GPU host (see docs/runbook_gpu.md, step 2).
#
# Usage:  VLLM_VERSION=<pinned version> bash scripts/gpu_setup.sh
#
# The script stops at the first error and prints each step. It never chooses a vLLM version itself.
set -euo pipefail

step() { printf '\n==> %s\n' "$*"; }

if [[ -z "${VLLM_VERSION:-}" ]]; then
  echo "error: set VLLM_VERSION to the vLLM version to pin (for example VLLM_VERSION=0.x.y)" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

step "install uv (if missing)"
if ! command -v uv >/dev/null 2>&1; then
  python3 -m pip install --user --quiet uv
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

step "create the virtual environment (.venv, Python 3.11 or newer)"
uv venv --python 3.11 .venv || uv venv .venv
PY=".venv/bin/python"

step "install vLLM ${VLLM_VERSION}"
uv pip install -p "$PY" "vllm==${VLLM_VERSION}"

step "install day0 in editable mode with development extras"
uv pip install -p "$PY" -e ".[dev]"

step "check the GPU"
nvidia-smi

step "check the vLLM version"
.venv/bin/vllm --version

step "check the day0 version"
.venv/bin/day0 --version

echo
echo "Setup complete. Activate with: source .venv/bin/activate"
