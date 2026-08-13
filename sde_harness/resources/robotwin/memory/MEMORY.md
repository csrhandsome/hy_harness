# RoboTwin reviewed memory index

This directory is a read-only runtime knowledge base. Workers may read it and
record episode-specific findings in their output audit, but must not edit it
during evaluation. Promote only reviewed, reproducible findings here offline.

Read only the notes relevant to the current task:

- [Control and chunk semantics](control_and_chunk_semantics.md): fresh-forward
  guarantees, prefix selection, cache invalidation, and when to use primitives.
- [Bimanual EE conventions](bimanual_ee_conventions.md): 16-value state/action
  layout, quaternion ordering, and conservative primitive use.
- [Failure and audit protocol](failure_and_audit_protocol.md): authoritative
  success, recovery discipline, and what an episode audit must preserve.

Task-specific calibrated coordinates and successful seed references belong
under `resources/robotwin/references/` after simulator validation.
