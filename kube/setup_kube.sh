#!/usr/bin/env bash

# Install kubectl when needed and validate access to an existing Kubernetes cluster.

set -euo pipefail

KUBECONFIG_PATH="${KUBECONFIG:-${HOME}/.kube/config}"
KUBECTL_INSTALL_DIR="${KUBECTL_INSTALL_DIR:-/usr/local/bin}"
KUBECTL_DOWNLOAD_PATH=""

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    printf 'Usage: bash %s\nValidate an existing cluster using KUBECONFIG. Install kubectl if missing (override KUBECTL_INSTALL_DIR).\n' "$0"
    exit 0
fi
[[ $# -eq 0 ]] || { printf 'Unexpected argument: %s\n' "$1" >&2; exit 1; }

cleanup() {
    if [[ -n "${KUBECTL_DOWNLOAD_PATH}" && -f "${KUBECTL_DOWNLOAD_PATH}" ]]; then
        rm -f -- "${KUBECTL_DOWNLOAD_PATH}"
    fi
}

trap cleanup EXIT

log() {
    printf '[kubernetes-setup] %s\n' "$*"
}

fail() {
    printf '[kubernetes-setup] ERROR: %s\n' "$*" >&2
    exit 1
}

install_kubectl() {
    local arch
    local kubectl_os
    local kubectl_arch
    local version

    command -v curl >/dev/null 2>&1 || fail "curl is required to install kubectl."

    case "$(uname -s)" in
        Linux) kubectl_os="linux" ;;
        Darwin) kubectl_os="darwin" ;;
        *) fail "Unsupported operating system: $(uname -s)" ;;
    esac

    arch="$(uname -m)"
    case "${arch}" in
        x86_64|amd64) kubectl_arch="amd64" ;;
        aarch64|arm64) kubectl_arch="arm64" ;;
        *) fail "Unsupported architecture: ${arch}" ;;
    esac

    version="$(curl -fsSL https://dl.k8s.io/release/stable.txt)"
    KUBECTL_DOWNLOAD_PATH="$(mktemp "${TMPDIR:-/tmp}/kubectl.XXXXXX")"

    log "Downloading kubectl ${version} for ${kubectl_os}/${kubectl_arch}"
    curl -fsSL -o "${KUBECTL_DOWNLOAD_PATH}" \
        "https://dl.k8s.io/release/${version}/bin/${kubectl_os}/${kubectl_arch}/kubectl"
    chmod 0755 "${KUBECTL_DOWNLOAD_PATH}"

    if [[ -w "${KUBECTL_INSTALL_DIR}" ]]; then
        install -m 0755 "${KUBECTL_DOWNLOAD_PATH}" "${KUBECTL_INSTALL_DIR}/kubectl"
    elif command -v sudo >/dev/null 2>&1; then
        sudo install -m 0755 "${KUBECTL_DOWNLOAD_PATH}" "${KUBECTL_INSTALL_DIR}/kubectl"
    else
        fail "Cannot write to ${KUBECTL_INSTALL_DIR}; set KUBECTL_INSTALL_DIR to a writable directory."
    fi
    export PATH="${KUBECTL_INSTALL_DIR}:${PATH}"
}

if ! command -v kubectl >/dev/null 2>&1; then
    install_kubectl
fi

[[ -f "${KUBECONFIG_PATH}" ]] || fail \
    "No kubeconfig found at ${KUBECONFIG_PATH}. Set KUBECONFIG to your cluster credentials."

chmod 600 "${KUBECONFIG_PATH}" || true
export KUBECONFIG="${KUBECONFIG_PATH}"

log "Using kubectl at $(command -v kubectl)"
kubectl version --client

log "Using context $(kubectl config current-context)"
kubectl cluster-info
kubectl get nodes

log "Kubernetes access is ready. Run python kube/test_kube.py for the Python client check."
