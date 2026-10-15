#!/usr/bin/env python3

"""Validate Kubernetes API access for the R2E-Gym namespace."""

import argparse


def test_connection(config_file: str | None, namespace: str) -> bool:
    from kubernetes import client, config

    try:
        if config_file:
            config.load_kube_config(config_file=config_file)
            print(f"Loaded kubeconfig: {config_file}")
        else:
            config.load_kube_config()
            print("Loaded kubeconfig from KUBECONFIG or ~/.kube/config")
    except Exception as exc:
        print(f"Failed to load kubeconfig: {exc}")
        return False

    try:
        version = client.VersionApi().get_code()
        print(f"Connected to Kubernetes {version.git_version}")
    except Exception as exc:
        print(f"Failed to connect to the Kubernetes API: {exc}")
        return False

    try:
        pods = client.CoreV1Api().list_namespaced_pod(namespace=namespace)
        print(f"Listed {len(pods.items)} pods in namespace {namespace!r}")
    except Exception as exc:
        print(f"Failed to list pods in namespace {namespace!r}: {exc}")
        return False

    print("Kubernetes API check passed")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "kubeconfig",
        nargs="?",
        help="Kubeconfig path; defaults to KUBECONFIG or ~/.kube/config",
    )
    parser.add_argument("--namespace", default="default")
    args = parser.parse_args()
    raise SystemExit(0 if test_connection(args.kubeconfig, args.namespace) else 1)


if __name__ == "__main__":
    main()
