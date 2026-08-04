"""RPC façade around a benchmark-specific action policy adapter."""

from __future__ import annotations

from typing import Any, Protocol


class PolicyAdapter(Protocol):
    benchmark: str

    def metadata(self) -> dict[str, Any]: ...

    def reset(self) -> Any: ...

    def dispatch(self, method: str, kwargs: dict[str, Any]) -> Any: ...


class PolicyRpcService:
    """Own one model instance and isolate its episode-local state by session."""

    def __init__(self, adapter: PolicyAdapter):
        self.adapter = adapter
        self._session_id: str | None = None

    def _activate(self, session_id: str | None) -> str:
        requested = str(session_id or "default")
        if self._session_id is None:
            self.adapter.reset()
            self._session_id = requested
        elif self._session_id != requested:
            raise RuntimeError(
                f"policy server is serving session {self._session_id!r}; "
                f"reset it before using {requested!r}"
            )
        return requested

    def dispatch(
        self, method: str, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        if args:
            raise TypeError("policy RPC methods accept keyword arguments only")
        if method == "metadata":
            return {
                "protocol_version": 1,
                "benchmark": self.adapter.benchmark,
                "active_session": self._session_id,
                **self.adapter.metadata(),
            }
        if method == "reset":
            session_id = str(kwargs.get("session_id") or "default")
            result = self.adapter.reset()
            self._session_id = session_id
            return {"ok": True, "session_id": session_id, "result": result}
        if method == "close":
            requested = str(kwargs.get("session_id") or "default")
            if self._session_id not in (None, requested):
                raise RuntimeError(
                    f"cannot close session {requested!r}; active={self._session_id!r}"
                )
            self.adapter.reset()
            self._session_id = None
            return {"ok": True, "session_id": requested}

        self._activate(kwargs.pop("session_id", None))
        return self.adapter.dispatch(method, kwargs)

