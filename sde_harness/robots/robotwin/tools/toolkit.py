"""Agent-facing RoboTwin toolkit and result serialization."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from hy_harness.tools import common
from hy_harness.tools.base import BaseTool
from hy_harness.utils.config import get_repo_root
from hy_harness.utils.logging import get_output_dir

from .adapter import RobotTwinEnvAdapter
from .execution import ExecutionResult, RobotTwinActionExecutor
from .tools_description import build_tools_description


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _png_bytes(image: Any) -> bytes | None:
    if image is None:
        return None
    arr = np.asarray(image)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] not in (1, 3, 4) and arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim != 3 or arr.shape[-1] not in (1, 3, 4):
        return None
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if np.issubdtype(arr.dtype, np.floating):
        if arr.size and float(np.nanmax(arr)) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0.0, 255.0)
    buffer = io.BytesIO()
    imageio.imwrite(buffer, np.ascontiguousarray(arr.astype(np.uint8)), format="png")
    return buffer.getvalue()


def _first_image(batch: dict[str, Any], *keys: str) -> Any:
    return next(
        (batch[key] for key in keys if key in batch and batch[key] is not None), None
    )


class RobotTwinTools(BaseTool):
    """Planner tools for one live RoboTwin episode."""

    def __init__(
        self,
        *,
        env: RobotTwinEnvAdapter,
        policy: Any | None = None,
        output_dir: str | Path | None = None,
        audit_path: str | Path | None = None,
        read_roots: list[str | Path] | None = None,
        dashboard: Any = None,
    ) -> None:
        super().__init__(dashboard=dashboard)
        self.env = env
        self.policy = policy
        self.output_dir = Path(output_dir).resolve() if output_dir else None
        self.audit_path = Path(audit_path).resolve() if audit_path else None
        roots = list(read_roots or [])
        if self.output_dir is not None:
            roots.append(self.output_dir)
        self.read_roots = tuple(Path(root).resolve() for root in roots)
        self._records: list[dict[str, Any]] = []
        self._call_counts: dict[str, int] = {}
        self._read_paths: list[str] = []
        self._customize_file_tools()
        self._restrict_finish()
        self._executor = RobotTwinActionExecutor(
            env=env, policy=policy, record=self._record
        )
        chunk_size = int(getattr(policy, "action_chunk_size", 50) or 50)
        default_steps = int(getattr(policy, "default_execute_steps", 30) or 30)
        handlers = {
            "robotwin_observe": self.observe,
            "robotwin_status": self.status,
            "robotwin_vla_chunk": self.vla_chunk,
            "robotwin_move_arm": self.move_arm,
            "robotwin_translate_arm": self.translate_arm,
            "robotwin_move_bimanual": self.move_bimanual,
            "robotwin_rotate_arm": self.rotate_arm,
            "robotwin_set_gripper": self.set_gripper,
            "robotwin_release": self.release,
            "robotwin_hold": self.hold,
            "robotwin_execute_ee": self.execute_ee,
        }
        for description in build_tools_description(
            model_chunk_size=chunk_size, default_execute_steps=default_steps
        ):
            self.add_tool(
                description["name"], description, handlers[description["name"]]
            )

    def _customize_file_tools(self) -> None:
        """Make memory reads auditable and restrict writes to the run output."""
        read_spec = {
            "name": "read_text_file",
            "description": (
                "Read a UTF-8 RoboTwin memory, reference, recipe, or audit file. Memory and "
                "references are read-only; relevant paths are recorded in the run summary."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "max_chars": {"type": "integer", "default": 40000},
                },
                "required": ["path"],
            },
        }
        list_spec = {
            "name": "list_dir",
            "description": (
                "List one directory to discover RoboTwin memory leaves, matching reviewed "
                "references, or current run artifacts."
            ),
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        }
        self.add_tool("read_text_file", read_spec, self.read_text_file)
        self.add_tool("list_dir", list_spec, self.list_dir)

        spec = {
            "name": "write_text_file",
            "description": (
                "Write the final audit or another run artifact under this episode output_dir. "
                "Writes outside output_dir are rejected; global memory is read-only."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        }
        self.add_tool("write_text_file", spec, self.write_output_text)

    def _restrict_finish(self) -> None:
        spec = {
            "name": "finish",
            "description": (
                "End the planner loop. Benchmark success is authoritative: a requested success is "
                "rejected unless robotwin_status reports success=true. Write the audit first."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "status": {"type": "string"},
                    "summary": {"type": "string"},
                },
                "required": ["status", "summary"],
            },
        }
        self.add_tool("finish", spec, self.finish)

    def finish(self, status: str, summary: str) -> dict[str, Any]:
        if self.audit_path is not None and not self.audit_path.is_file():
            return {
                "error": f"write the required audit before finish: {self.audit_path}",
                **self.env.status(),
            }
        benchmark = self.env.status()
        if benchmark["success"]:
            return {
                "_finish": True,
                "status": "success",
                "summary": summary,
                **benchmark,
            }
        requested = str(status).strip().lower()
        if requested == "success":
            return {
                "error": "benchmark success is false; continue or finish as failure/stuck",
                **benchmark,
            }
        return {
            "_finish": True,
            "status": requested or "failure",
            "summary": summary,
            **benchmark,
        }

    def _resolve_read_path(self, path: str) -> Path:
        target = Path(path).expanduser()
        if not target.is_absolute():
            target = get_repo_root() / target
        return target.resolve()

    def _read_allowed(self, target: Path) -> bool:
        return not self.read_roots or any(
            target == root or root in target.parents for root in self.read_roots
        )

    def read_text_file(self, path: str, max_chars: int = 40000) -> dict[str, Any]:
        target = self._resolve_read_path(path)
        if not self._read_allowed(target):
            return {
                "error": f"reads are restricted to {list(map(str, self.read_roots))}"
            }
        result = common.read_text_file(str(target), max_chars=max_chars)
        resolved = result.get("path")
        if resolved and resolved not in self._read_paths:
            self._read_paths.append(str(resolved))
        return result

    def list_dir(self, path: str = "") -> dict[str, Any]:
        target = self.output_dir if not path else self._resolve_read_path(path)
        if target is None:
            return {"error": "no default output_dir is configured"}
        if not self._read_allowed(target):
            return {
                "error": f"reads are restricted to {list(map(str, self.read_roots))}"
            }
        return common.list_dir(str(target))

    def write_output_text(self, path: str, content: str) -> dict[str, Any]:
        if self.output_dir is None:
            return {"error": "RoboTwin output_dir is not configured"}
        target = Path(path).expanduser()
        if not target.is_absolute():
            target = self.output_dir / target
        target = target.resolve()
        if target != self.output_dir and self.output_dir not in target.parents:
            return {"error": f"writes are restricted to {self.output_dir}"}
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {"path": str(target), "bytes_written": len(content.encode("utf-8"))}

    def _payload(
        self, *, include_images: bool = True, result: ExecutionResult | None = None
    ) -> dict[str, Any]:
        batch = self.env.encoded_observation()
        payload: dict[str, Any] = {
            **self.env.status(),
            "state": _jsonable(batch.get("observation.state")),
        }
        if result is not None:
            payload.update(
                applied_steps=result.applied_steps,
                last_action=_jsonable(result.last_action),
                forward_count=result.forward_count,
                generated_actions=result.generated_actions,
                discarded_actions=result.discarded_actions,
                chunk_id=result.chunk_id,
            )
        if include_images:
            payload["_image_bytes"] = _png_bytes(
                _first_image(
                    batch, "raw_images.top_head", "observation.images.top_head"
                )
            )
            payload["_image_cam_bytes"] = _png_bytes(
                _first_image(
                    batch, "raw_images.hand_left", "observation.images.hand_left"
                )
            )
            payload["_image_wrist_bytes"] = _png_bytes(
                _first_image(
                    batch, "raw_images.hand_right", "observation.images.hand_right"
                )
            )
        return payload

    def observe(self) -> dict[str, Any]:
        self.env.refresh()
        return self._payload()

    def status(self) -> dict[str, Any]:
        return self.env.status()

    def _run(self, method: str, **kwargs: Any) -> dict[str, Any]:
        self._call_counts[method] = self._call_counts.get(method, 0) + 1
        return self._payload(result=getattr(self._executor, method)(**kwargs))

    def artifact_summary(self) -> dict[str, Any]:
        return {
            "tool_call_counts": dict(sorted(self._call_counts.items())),
            "files_read": list(self._read_paths),
            "vla_chunks_used": self._call_counts.get("vla_chunk", 0),
            "primitive_calls_used": sum(
                count
                for name, count in self._call_counts.items()
                if name != "vla_chunk"
            ),
            "simulator_actions": len(self._records),
            "vla_actions": sum(
                record["source"] == "robotwin_vla_chunk" for record in self._records
            ),
            "primitive_actions": sum(
                record["source"] != "robotwin_vla_chunk" for record in self._records
            ),
        }

    def vla_chunk(
        self,
        execute_steps: int | None = None,
        instruction_override: str | None = None,
    ) -> dict[str, Any]:
        if execute_steps is None:
            execute_steps = int(getattr(self.policy, "default_execute_steps", 30) or 30)
        return self._run(
            "vla_chunk",
            execute_steps=execute_steps,
            instruction_override=instruction_override,
        )

    def move_arm(self, **kwargs: Any) -> dict[str, Any]:
        return self._run("move_arm", **kwargs)

    def translate_arm(self, **kwargs: Any) -> dict[str, Any]:
        return self._run("translate_arm", **kwargs)

    def move_bimanual(self, **kwargs: Any) -> dict[str, Any]:
        return self._run("move_bimanual", **kwargs)

    def rotate_arm(self, **kwargs: Any) -> dict[str, Any]:
        return self._run("rotate_arm", **kwargs)

    def set_gripper(self, **kwargs: Any) -> dict[str, Any]:
        return self._run("set_gripper", **kwargs)

    def release(self, **kwargs: Any) -> dict[str, Any]:
        return self._run("release", **kwargs)

    def hold(self, **kwargs: Any) -> dict[str, Any]:
        return self._run("hold", **kwargs)

    def execute_ee(self, action: list[float], repeat: int = 1) -> dict[str, Any]:
        return self._run("execute_ee", action=action, repeat=repeat)

    def _record(self, action: Any, *, action_type: str, source: str) -> None:
        self._records.append(
            {
                "step": len(self._records),
                "source": source,
                "action_type": action_type,
                "action": _jsonable(action),
                **self.env.status(),
            }
        )

    def write_recipe(self, recipe_tag: str) -> str:
        output_dir = self.output_dir or get_output_dir() or Path.cwd()
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"recipe_{recipe_tag}.jsonl"
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for record in self._records:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        temporary.replace(path)
        return str(path)


__all__ = ["RobotTwinTools"]
