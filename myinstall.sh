#!/usr/bin/env bash

# Install the GPU training environment used by this repository.
# Run this script from any directory on a Linux machine with NVIDIA drivers.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${BSR_VENV_DIR:-${RLLM_VENV_DIR:-${REPO_ROOT}/.venv}}"
PYTHON_VERSION="${BSR_PYTHON_VERSION:-${RLLM_PYTHON_VERSION:-3.10.14}}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    printf 'Usage: bash %s\nLinux NVIDIA/CUDA installation. Override BSR_VENV_DIR or BSR_PYTHON_VERSION as needed.\n' "$0"
    exit 0
fi
[[ $# -eq 0 ]] || { printf 'Unexpected argument: %s\n' "$1" >&2; exit 1; }

log() {
    printf '[gpu-setup] %s\n' "$*"
}

fail() {
    printf '[gpu-setup] ERROR: %s\n' "$*" >&2
    exit 1
}

if [[ "$(uname -s)" != "Linux" ]]; then
    fail "This training environment is supported on Linux GPU machines."
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    fail "nvidia-smi was not found. Install a compatible NVIDIA driver first."
fi

if ! command -v nvcc >/dev/null 2>&1; then
    fail "nvcc was not found. Install a CUDA toolkit before building FlashAttention."
fi

if ! command -v curl >/dev/null 2>&1; then
    fail "curl is required to install uv."
fi

if ! command -v uv >/dev/null 2>&1; then
    log "Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    UV_BIN="${HOME}/.local/bin/uv"
    [[ -x "${UV_BIN}" ]] || fail "uv installation completed, but ${UV_BIN} was not found."
else
    UV_BIN="$(command -v uv)"
fi

log "Creating Python ${PYTHON_VERSION} environment at ${VENV_DIR}"
"${UV_BIN}" venv --python "${PYTHON_VERSION}" "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# Apply the same constraints to every install so editable dependencies cannot
# silently replace the GPU stack required by the bundled vLLM patches.
export UV_CONSTRAINT="${REPO_ROOT}/requirements-gpu.txt"

cd "${REPO_ROOT}"

log "Installing the recorded GPU stack"
"${UV_BIN}" pip install setuptools wheel build packaging ninja
"${UV_BIN}" pip install \
    torch==2.7.0 \
    vllm==0.9.2 \
    ray==2.47.1 \
    transformers==4.53.2 \
    datasets==2.19.0

MAX_JOBS="${MAX_JOBS:-$(nproc)}" UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}" \
    "${UV_BIN}" pip install flash-attn==2.8.1 --no-build-isolation

log "Installing BSR and the bundled veRL and R2E-Gym packages"
# vLLM is installed separately above because this veRL snapshot's optional
# vllm extra still constrains vLLM to an older version.
"${UV_BIN}" pip install -e "${REPO_ROOT}/verl"
"${UV_BIN}" pip install -e "${REPO_ROOT}/R2E-Gym"
"${UV_BIN}" pip install -e "${REPO_ROOT}[gpu]"

log "Installing runtime compatibility dependencies"
"${UV_BIN}" pip install \
    azure-identity \
    antlr4-python3-runtime==4.9.3 \
    debugpy \
    hf_transfer \
    protobuf==4.25.8 \
    e2b==1.3.3 \
    e2b-code-interpreter==1.2.0 \
    seaborn \
    nvitop \
    wandb==0.21.0

# Avoid conflicts between the legacy pynvml packages and nvidia-ml-py used by nvitop.
"${UV_BIN}" pip uninstall nvidia-ml-py3 pynvml
"${UV_BIN}" pip install --upgrade nvidia-ml-py nvitop
"${UV_BIN}" pip check

log "Applying the repository's vLLM 0.9.2 patches"
VLLM_ROOT="$(python - <<'PY'
from pathlib import Path
import vllm

print(Path(vllm.__file__).resolve().parent)
PY
)"

python - <<'PY'
import vllm

if vllm.__version__ != "0.9.2":
    raise SystemExit(f"Expected vLLM 0.9.2 before patching, found {vllm.__version__}")
PY

install -m 0644 "${REPO_ROOT}/vllm_patch/scheduler.py" "${VLLM_ROOT}/v1/core/sched/scheduler.py"
install -m 0644 "${REPO_ROOT}/vllm_patch/gpu_worker.py" "${VLLM_ROOT}/v1/worker/gpu_worker.py"
install -m 0644 "${REPO_ROOT}/vllm_patch/block_pool.py" "${VLLM_ROOT}/v1/core/block_pool.py"

log "Checking the installation"
python - <<'PY'
import datasets
import ray
import torch
import transformers
import vllm

assert torch.cuda.is_available(), "PyTorch cannot see a CUDA GPU"
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
print(f"vllm={vllm.__version__}")
print(f"ray={ray.__version__}")
print(f"transformers={transformers.__version__}")
print(f"datasets={datasets.__version__}")
print(f"gpus={torch.cuda.device_count()}")
PY

log "Setup complete. Activate it with: source ${VENV_DIR}/bin/activate"
log "Before training, export HF_TOKEN, WANDB_API_KEY, and KUBECONFIG."
log "Kubernetes instructions: ${REPO_ROOT}/kube/README.md"
