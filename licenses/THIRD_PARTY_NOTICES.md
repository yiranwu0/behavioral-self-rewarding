# Third-party notices

Behavioral Self-Rewarding is distributed under [Apache License 2.0](../LICENSE).
Existing upstream copyright and license headers remain applicable to the
files containing them. BSR's authorship does not replace upstream authorship.

| Component | Origin | License and retained notices |
| --- | --- | --- |
| `rllm/` and upstream examples | [rLLM / Agentica Project](https://github.com/agentica-project/rllm), including DeepSWE | Apache-2.0; [LICENSE](../LICENSE), original source headers |
| `R2E-Gym/` | [R2E-Gym](https://github.com/R2E-Gym/R2E-Gym) | Apache-2.0; [R2E-Gym/LICENSE](../R2E-Gym/LICENSE), original source headers |
| `verl/` and adapted trainer components | [veRL / ByteDance](https://github.com/volcengine/verl) | Apache-2.0; [verl/LICENSE](../verl/LICENSE), original source headers |
| `vllm_patch/` | [vLLM](https://github.com/vllm-project/vllm) | Apache-2.0; [LICENSE](../LICENSE), SPDX copyright headers on each patch |
| `rllm/rewards/code_utils/pyext2.py` | Ryan Gonzalez, pyext | MIT; complete copyright and permission notice retained in [the source](../rllm/rewards/code_utils/pyext2.py) |

BSR modifications include pairwise behavioral judging, weighted
Bradley–Terry aggregation and bounded reward refinement, trajectory
compaction, rollout/trainer integration, experiment launchers, and portable
setup/evaluation helpers. Modified files retain their upstream notices.

Installed packages, model weights, datasets, and task container images have
their own licenses and are not relicensed by this repository. The public
distribution name is `behavioral-self-rewarding`; `rllm` remains the Python
import namespace inherited from the training framework.
