"""clevr_langchain — drop-in LangChain integration for Clevr.

LangChain agents declare tools via `@tool` decorators or BaseTool subclasses.
This module wraps any callable tool so EVERY invocation routes through
/v1/evaluate before the underlying function runs.

Two integration styles, depending on how the LangChain code is structured:

1. **Wrap a tool function** — the most direct path:

       from langchain_core.tools import tool
       from clevr import Clevr
       from clevr_langchain import guarded

       clevr = Clevr(agent="researcher")

       @guarded(clevr, tool_name="kb_search", action_type="read")
       @tool
       def kb_search(query: str) -> str:
           return my_kb.search(query)

2. **Callback handler** — for code paths where tools come from elsewhere
   (LangGraph, agent executors). Hooks into LangChain's BaseCallbackHandler:

       from langchain_core.callbacks import BaseCallbackHandler
       from clevr_langchain import ClevrCallbackHandler

       handler = ClevrCallbackHandler(clevr)
       agent_executor.invoke({"input": ...}, {"callbacks": [handler]})

The handler captures `on_tool_start`/`on_tool_end` events and calls evaluate
on each. Block/escalate verdicts surface back as tool errors that the LLM
can react to. In shadow mode, decisions are recorded but tools always run.
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Optional

from clevr import Clevr, ClevrBlockedError, ClevrEscalatedError


def guarded(clevr: Clevr, tool_name: str, action_type: str = "tool_call",
            target_of: Optional[Callable[[dict], Optional[str]]] = None,
            environment: Optional[str] = None) -> Callable:
    """Decorator factory wrapping a LangChain @tool with Clevr.guard.

    Args:
        clevr: Configured Clevr client.
        tool_name: Logical tool id surfaced in the audit log.
        action_type: 'read' | 'write' | 'delete' | 'exec' | 'send' | 'tool_call'.
        target_of: Optional callable that turns the tool's kwargs into a target
                   string for policy matching (e.g. f"kb/{kwargs['query']}").
        environment: 'prod' | 'staging' | 'dev' (default None).
    """
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            target = None
            if target_of:
                try:
                    target = target_of(kwargs)
                except Exception:
                    target = None
            action = {
                "tool": tool_name,
                # The platform, unless the client names one (a LangGraph agent
                # sets runtime="langgraph" on its client: this adapter cannot tell).
                "runtime": getattr(clevr, "runtime", None) or "langchain",
                "action_type": action_type,
                "action": f"{tool_name}({_short(kwargs)})",
                "target": target,
                "environment": environment,
                "metadata": {"args": _short(args), "kwargs": _short(kwargs)},
            }
            return clevr.guard(action, run=lambda _v: fn(*args, **kwargs))
        return wrapper
    return decorator


class ClevrCallbackHandler:
    """LangChain BaseCallbackHandler-compatible interceptor.

    Use this when you can't easily decorate each tool — e.g. when LangChain's
    AgentExecutor pulls tools from a registry. Hook it as a callback and
    every tool invocation is guarded transparently.

    Implementation note: we don't subclass BaseCallbackHandler at import time
    because we don't want clevr.py to require langchain as a dependency.
    Users instantiate this class explicitly; it duck-types the callback API.
    """

    def __init__(self, clevr: Clevr, action_type: str = "tool_call") -> None:
        self.clevr = clevr
        self.action_type = action_type
        self._verdicts: dict[str, dict] = {}  # run_id → verdict

    def on_tool_start(self, serialized: dict, input_str: str, *, run_id: Any, **kwargs: Any) -> None:
        tool_name = (serialized or {}).get("name") or "unknown_tool"
        verdict = self.clevr.evaluate({
            "tool": tool_name,
            "runtime": getattr(self.clevr, "runtime", None) or "langchain",
            "action_type": self.action_type,
            "action": f"{tool_name}({_truncate(input_str, 200)})",
            "metadata": {"input": _truncate(input_str, 500)},
        })
        self._verdicts[str(run_id)] = verdict
        if self.clevr.mode == "enforce":
            effect = verdict.get("effect")
            if effect == "block":
                raise ClevrBlockedError(verdict)
            # 'step_up' is the legacy synonym for a held/escalate verdict; without
            # it a step_up tool call would run instead of being held.
            if effect in ("escalate", "step_up") and self.clevr.on_escalate == "throw":
                raise ClevrEscalatedError(verdict)
        # shadow mode: let the tool run, retain verdict for on_tool_end

    def on_tool_end(self, output: str, *, run_id: Any, **kwargs: Any) -> None:
        verdict = self._verdicts.pop(str(run_id), None)
        if verdict and self.clevr.mode == "shadow" and verdict.get("effect") != "allow":
            # Caller might log this if they want; we keep it on the client
            # via clevr.last_verdict so the next code line can inspect.
            self.clevr.last_verdict = verdict

    def on_tool_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
        self._verdicts.pop(str(run_id), None)


def _short(obj: Any) -> str:
    try:
        import json
        return json.dumps(obj, default=str)[:200]
    except Exception:
        return str(obj)[:200]


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"
