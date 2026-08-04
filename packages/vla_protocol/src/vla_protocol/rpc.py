"""Small HTTP/JSON RPC transport with lossless NumPy array support."""

from __future__ import annotations

import base64
import json
import threading
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

import numpy as np


DEFAULT_TIMEOUT_S = 30.0


class RpcError(RuntimeError):
    """Remote call failure with the server traceback attached when available."""

    def __init__(self, method: str, message: str, traceback: str | None = None):
        super().__init__(f"RPC {method!r} failed: {message}")
        self.method = method
        self.remote_traceback = traceback


def _from_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        if "__ndarray__" in obj and set(obj) <= {"__ndarray__", "dtype", "shape"}:
            raw = base64.b64decode(obj["__ndarray__"])
            return np.frombuffer(raw, dtype=obj["dtype"]).reshape(obj["shape"]).copy()
        return {key: _from_json(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_from_json(value) for value in obj]
    return obj


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return {
                "__ndarray__": base64.b64encode(obj.tobytes()).decode("ascii"),
                "dtype": str(obj.dtype),
                "shape": list(obj.shape),
            }
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


class RpcClient:
    """Synchronous client for a :class:`RpcServer`."""

    def __init__(self, base_url: str, *, timeout_s: float = DEFAULT_TIMEOUT_S):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)

    def call(
        self,
        method: str,
        *,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> Any:
        request_id = str(uuid.uuid4())
        body = json.dumps(
            {
                "id": request_id,
                "method": method,
                "args": list(args),
                "kwargs": kwargs or {},
            },
            cls=_NumpyEncoder,
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/call",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout_s if timeout_s is None else timeout_s
            ) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            raw = exc.read()
        except OSError as exc:
            raise RpcError(method, f"HTTP request failed: {exc}") from exc

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RpcError(method, f"invalid JSON response: {exc}") from exc
        if not isinstance(payload, dict) or not payload.get("ok"):
            detail = payload.get("error", "invalid response") if isinstance(payload, dict) else "invalid response"
            traceback = payload.get("traceback") if isinstance(payload, dict) else None
            raise RpcError(method, str(detail), traceback=traceback)
        if payload.get("id") != request_id:
            raise RpcError(method, "response id mismatch")
        return _from_json(payload.get("result"))

    def healthz(self, *, timeout_s: float = 2.0) -> dict[str, Any]:
        return self.call("healthz", timeout_s=timeout_s)

    def close(self) -> None:
        return None


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def do_POST(self) -> None:
        if self.path != "/call":
            self.send_response(404)
            self.end_headers()
            return
        request_id = None
        try:
            size = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(size))
            request_id = request.get("id")
            method = request["method"]
            args = tuple(_from_json(value) for value in request.get("args", []))
            kwargs = {
                key: _from_json(value)
                for key, value in request.get("kwargs", {}).items()
            }
            result = self.server.dispatch(method, args, kwargs)  # type: ignore[attr-defined]
            payload = {"id": request_id, "ok": True, "result": result}
        except Exception as exc:
            import traceback

            payload = {
                "id": request_id,
                "ok": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        body = json.dumps(payload, cls=_NumpyEncoder).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class RpcServer(ThreadingHTTPServer):
    """Threaded RPC server that serializes model calls with one lock."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        dispatch: Callable[[str, tuple[Any, ...], dict[str, Any]], Any],
    ) -> None:
        super().__init__(address, _Handler)
        self._dispatch = dispatch
        self._dispatch_lock = threading.Lock()

    def dispatch(
        self, method: str, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        if method == "healthz":
            return {"status": "ok"}
        with self._dispatch_lock:
            return self._dispatch(method, args, kwargs)

