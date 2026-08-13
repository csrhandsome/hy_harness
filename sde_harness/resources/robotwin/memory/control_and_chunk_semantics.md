# Control and chunk semantics

- `robotwin_vla_chunk` means one fresh model forward from the current state.
  It never consumes a suffix decoded from an earlier state.
- Hy-VLA RoboTwin checkpoints normally model a 50-action physical chunk. The
  deployment default executes a 30-action prefix; use shorter 1-5 action
  prefixes around contact, grasp, articulation, release, and final alignment.
- The unused suffix is intentionally discarded. A subsequent chunk starts a
  new forward from a new observation.
- Every deterministic primitive invalidates residual VLA cache before moving.
- Prefer a coherent semantic primitive that executes several interpolated
  simulator actions over one planner decision per simulator action.
- Motion tool results already contain post-action state and camera images.
  Re-observe only when those images are stale or insufficient.
