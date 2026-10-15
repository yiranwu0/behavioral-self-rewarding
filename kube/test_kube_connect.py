#!/usr/bin/env python3

"""Create one R2E-Gym Kubernetes environment and execute a command in it."""

import argparse
import time
from pathlib import Path



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="R2E-Gym/R2E-Gym-Subset")
    parser.add_argument("--split", default="train")
    parser.add_argument("--index", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    import r2egym
    from datasets import load_dataset
    from r2egym.agenthub.action import Action
    from r2egym.agenthub.environment.env import EnvArgs, RepoEnv

    dataset = load_dataset(args.dataset, split=args.split)
    if not 0 <= args.index < len(dataset):
        raise SystemExit(f"--index must be between 0 and {len(dataset) - 1}")

    entry = dataset[args.index]
    image = entry.get("docker_image", "<unknown>")
    print(f"Creating Kubernetes environment for {image}", flush=True)

    tool_path = (
        Path(r2egym.__file__).resolve().parent
        / "agenthub"
        / "tools"
        / "execute_bash.py"
    )
    if not tool_path.is_file():
        raise SystemExit(f"R2E-Gym execute_bash tool not found: {tool_path}")

    environment = None
    try:
        started = time.monotonic()
        environment = RepoEnv(
            EnvArgs(ds=entry),
            backend="kubernetes",
            verbose=True,
            pod_tag="kube-smoke-test",
        )
        environment.add_commands([str(tool_path)])
        result = environment.run_action(
            Action(
                function_name="execute_bash",
                parameters={"command": "pwd && test -d /testbed"},
            ),
            timeout=args.timeout,
        )
        print(result[0], flush=True)
        if result[1] != 0:
            raise RuntimeError(
                f"Smoke-test command failed with exit code {result[1]}"
            )
        print(
            f"Environment command completed in {time.monotonic() - started:.2f}s",
            flush=True,
        )
    finally:
        if environment is not None:
            print("Deleting smoke-test pod", flush=True)
            environment.close()


if __name__ == "__main__":
    main()
