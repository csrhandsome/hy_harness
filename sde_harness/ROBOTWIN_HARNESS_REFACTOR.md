# RoboTwin Harness refactor record

Date: 2026-08-13

## What changed

- Split the old monolithic `robots/robotwin/tools.py` into `tools/adapter.py`,
  `tools/execution.py`, `tools/tools_description.py`, and `tools/toolkit.py`.
- Added semantic dual-arm tools: absolute/relative single-arm motion,
  synchronized bimanual motion, rotation, gripper set/release, hold, plus a raw
  16-value EE debug escape hatch.
- Replaced `robotwin_vla_step` with `robotwin_vla_chunk`. One call now means
  exactly one fresh forward from the current state, followed by the requested
  prefix. Old cache and the unused suffix are discarded. Results report model
  forwards, generated/executed/discarded actions, and chunk id.
- Extended the policy RPC with `robotwin.chunk`, `robotwin.observe`, and
  `robotwin.invalidate_cache`; retained `robotwin.step` for direct evaluation.
  Every deterministic correction invalidates cached VLA actions.
- Set the deployment prefix to 30 actions. The model's physical chunk size is
  read from checkpoint/server metadata (normally 50 for RoboTwin).
- Added `primitive_first` and `vla_first` prompt profiles. Select with
  `harness.prompt_profile` or `ROBOTWIN_HARNESS_PROMPT_PROFILE`. Neither prompt
  enforces act/observe alternation.
- Made the vLLM tool-choice proxy profile-neutral: it requires a structured
  tool call but no longer forces observe or VLA as the next tool.
- Ported the RPent artifact flow: reviewed read-only indexed memory, scoped
  memory/reference reads, output-only writes, required episode audit, automatic
  physics-action recipe, and automatic fallback audit on planner failure.
  CodeBuddy built-in Bash/Read/Write/Glob/Grep tools are disabled by default;
  the scoped MCP tools remain available.
- Benchmark `success=true` is now required before `finish(status="success")`
  can terminate the planner.

## Verification

- Ruff format and static checks: passed.
- Python compile check for all changed modules: passed.
- `sde_harness` RoboTwin tool/prompt/memory/proxy tests: 13 passed.
- Policy wrapper/RPC/adapter/cache tests: 5 passed.
- Total targeted tests: 18 passed.

The tests cover fresh-forward/cache invalidation, prefix execution and suffix
discard, environment-success early stop, primitive interpolation and inactive
arm preservation, gripper-only motion, scoped file IO, mandatory audit,
automatic recipe naming, both prompt profiles, profile-neutral proxy behavior,
and local/remote RPC shapes.

## Server validation still required

Restart both the updated policy servers and RoboTwin workers so the new RPC
methods are present on both sides. Run a small A/B with identical tasks/seeds
for `primitive_first` and `vla_first`, then compare benchmark success, VLA
forwards, generated/executed/discarded actions, primitive calls/actions, and
false success attempts.

Segmentation/back-projection tools were not added: the current RoboTwin harness
only exposes RGB and EE state, not a verified depth/intrinsics/extrinsics path.
Add those tools only after the server observation contract provides calibrated
geometry; otherwise absolute world-coordinate targets would be guesses.
