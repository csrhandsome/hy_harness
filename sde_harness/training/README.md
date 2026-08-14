# Critic Stack training

This directory owns the offline lifecycle of the two-level Critic Stack.
It does **not** train Hy-VLA, update a deployed critic, or select recovery
tools online.

| Entry point | Role | Main artifact |
| --- | --- | --- |
| `build_critic_dataset.py` | Convert baseline rollout/replay checkpoints into versioned examples and split labels. | Dataset manifest |
| `train_fast_critic.py` | Train the low-latency outcome-risk model. The current intended implementation is Agent-SAFE. | Frozen Fast Critic checkpoint |
| `train_semantic_critic.py` | Train or adapt the slower semantic judge for target/region/order drift. | Frozen Semantic Critic checkpoint or adapter |
| `calibrate_critic_stack.py` | Calibrate each model and choose deterministic Router thresholds/budgets on validation data. | Versioned stack configuration |
| `evaluate_critic_stack.py` | Report per-level and end-to-end metrics, including missed semantic drift and intervention cost. | Evaluation report |

The two critics are not weight-fused.  At deployment they remain separate
frozen components: Fast Critic produces conditional terminal-failure risk;
the deterministic Router decides whether Semantic Critic is needed; Semantic
Critic returns semantic-ok, drift, or abstain.  The stack configuration only
contains calibration and routing values.

All dataset records should preserve `episode_id`, `checkpoint_id`, environment
name, model/checkpoint versions, input-evidence references, counterfactual or
terminal outcome labels, and the split assignment.  Do not derive a training
label from a trajectory after a recovery intervention without recording that
intervention and its counterfactual policy.

