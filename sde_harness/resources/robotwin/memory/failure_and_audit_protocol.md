# Failure and audit protocol

- `success=true` from the RoboTwin benchmark is the only success authority.
  A visually plausible placement is not sufficient.
- Do not reset within an evaluated episode. Recover in place while the action
  budget safely permits it; otherwise write an honest unsuccessful audit.
- Before `finish`, write one JSON audit in the episode output directory with:
  task/test id, prompt profile, memory files read, strategy, outcome,
  benchmark success, final status, VLA chunk count, primitive call count, and
  failure reason when applicable.
- The runner exports the physics-only JSONL recipe automatically. Global
  memory remains unchanged until an offline reviewer validates and promotes a
  reusable lesson.
