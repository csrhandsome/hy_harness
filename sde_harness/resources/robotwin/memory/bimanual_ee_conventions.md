# Bimanual EE conventions

The native 16-value state and action layout is:

`left xyz (3), left quaternion wxyz (4), left gripper (1), right xyz (3), right quaternion wxyz (4), right gripper (1)`.

- Named arm tools hold the unspecified arm and unspecified pose fields at the
  current values.
- Use relative translations only for small, visually justified corrections.
  The tool rejects a single delta longer than 0.30 m.
- Gripper targets use RoboTwin's native numeric value. Do not assume LIBERO's
  `-1=open, +1=close` convention; infer valid values from observed state and
  validated task references.
- Use raw `robotwin_execute_ee` only as a debug escape hatch. Named primitives
  expose intent and generate more useful recipes.
