# vLLM patches

This directory contains the three vLLM source files used by the training code. They target **vLLM 0.9.2**:

- `scheduler.py`
- `gpu_worker.py`
- `block_pool.py`

The supported GPU setup script installs vLLM 0.9.2, locates it in the active virtual environment, checks the version, and applies these files automatically:

```bash
bash myinstall.sh
```

To verify an installed environment after setup:

```bash
source .venv/bin/activate

VLLM_ROOT="$(python -c 'from pathlib import Path; import vllm; print(Path(vllm.__file__).resolve().parent)')"

diff -u vllm_patch/scheduler.py "$VLLM_ROOT/v1/core/sched/scheduler.py"
diff -u vllm_patch/gpu_worker.py "$VLLM_ROOT/v1/worker/gpu_worker.py"
diff -u vllm_patch/block_pool.py "$VLLM_ROOT/v1/core/block_pool.py"
```

Restart Ray workers and vLLM processes after applying the patches to an existing environment.
