"""clevr_crewai — drop-in CrewAI integration for Clevr.

CrewAI organizes work into Agents that hold Tools and execute Tasks. Tools
are either:
    • subclasses of `crewai.tools.BaseTool` (the canonical path), OR
    • plain Python functions decorated with `@tool` from crewai.tools.

This module wraps both surfaces so EVERY tool invocation routes through
/v1/evaluate before the underlying implementation runs. We don't import
crewai at module load time — that keeps clevr-py free of optional deps.

Two integration styles:

1. **Wrap an existing BaseTool instance** (recommended):

       from crewai.tools import BaseTool
       from clevr import Clevr
       from clevr_crewai import guard

       class KbSearch(BaseTool):
           name = "kb_search"
           description = "Search the knowledge base"
           def _run(self, query: str) -> str:
               return my_kb.search(query)

       clevr = Clevr(agent="researcher")
       agent = Agent(
           role="Researcher",
           goal="...",
           tools=[guard(clevr, KbSearch())],
       )

2. **Decorator on a plain @tool function**:

       from crewai.tools import tool
       from clevr_crewai import guarded

       @guarded(clevr, action_type="read")
       @tool("kb_search")
       def kb_search(query: str) -> str:
           return my_kb.search(query)

Either way the audit chain captures the tool name, the agent's actor chain,
and the verdict — block, escalate, or allow.
"""

from __future__ import annotations

import json
from functools import wraps
from typing import Any, Callable, Optional

from clevr import Clevr, ClevrBlockedError, ClevrEscalatedError


def guard(
    clevr: Clevr,
    tool: Any,
    *,
    tool_name: Optional[str] = None,
    action_type: str = "tool_call",
    target_of: Optional[Callable[[dict], Optional[str]]] = None,
    environment: Optional[str] = None,
) -> Any:
    """Wrap a CrewAI BaseTool instance so its _run is gated by Clevr.

    Mutates the instance in place AND returns it for chaining. We monkey-patch
    `_run` rather than subclass to preserve CrewAI's introspection (name,
    description, args_schema).
    """
    name = tool_name or getattr(tool, "name", None) or "unknown_tool"
    original = getattr(tool, "_run", None)
    if not callable(original):
        raise TypeError(
            f"Clevr crewai: tool {name!r} has no callable _run() method"
        )

    def gated(self, *args, **kwargs):  # noqa: ANN001 — keep BaseTool's signature
        # CrewAI passes positional args into _run when the schema is single-arg
        # (Pydantic v1 style), and kwargs when it's multi-field (v2). We pass
        # a single dict to target_of so user code can read either style.
        payload = dict(kwargs)
        if args:
            payload["_args"] = list(args)
        target = None
        if target_of:
            try:
                target = target_of(payload)
            except Exception:
                target = None
        action = {
            "tool": name,
            "action_type": action_type,
            "action": f"{name}({_short(payload)})",
            "target": target,
            "environment": environment,
            "metadata": {"args": _short(args), "kwargs": _short(kwargs)},
        }
        try:
            return clevr.guard(action, run=lambda _v: original(*args, **kwargs))
        except ClevrBlockedError as e:
            # CrewAI tools surface errors back to the agent via the returned
            # string — same shape as the LangChain handler.
            return f"[Clevr blocked this action: {e.verdict.get('reason', '')}]"
        except ClevrEscalatedError as e:
            return (
                "[Clevr requires human approval before this action runs: "
                f"{e.verdict.get('reason', '')}]"
            )

    # Bind to the instance so `self` is correct.
    import types
    tool._run = types.MethodType(gated, tool)
    return tool


def guarded(
    clevr: Clevr,
    *,
    tool_name: Optional[str] = None,
    action_type: str = "tool_call",
    target_of: Optional[Callable[[dict], Optional[str]]] = None,
    environment: Optional[str] = None,
) -> Callable:
    """Decorator: wrap a @tool-decorated function with Clevr.guard.

    Apply this OUTSIDE crewai's @tool — i.e. `@guarded(...)` above `@tool(...)`.
    """
    def decorator(fn: Callable) -> Callable:
        name = tool_name or getattr(fn, "name", None) or fn.__name__

        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            target = None
            if target_of:
                try:
                    target = target_of(kwargs)
                except Exception:
                    target = None
            action = {
                "tool": name,
                "action_type": action_type,
                "action": f"{name}({_short(kwargs)})",
                "target": target,
                "environment": environment,
                "metadata": {"args": _short(args), "kwargs": _short(kwargs)},
            }
            try:
                return clevr.guard(action, run=lambda _v: fn(*args, **kwargs))
            except ClevrBlockedError as e:
                return f"[Clevr blocked this action: {e.verdict.get('reason', '')}]"
            except ClevrEscalatedError as e:
                return (
                    "[Clevr requires human approval before this action runs: "
                    f"{e.verdict.get('reason', '')}]"
                )

        return wrapper
    return decorator


def guard_all(clevr: Clevr, tools: list, *, meta_by_name: Optional[dict] = None) -> list:
    """Convenience: wrap every BaseTool instance in `tools` with `guard`."""
    meta_by_name = meta_by_name or {}
    out = []
    for t in tools:
        m = meta_by_name.get(getattr(t, "name", ""), {})
        out.append(guard(clevr, t, **m))
    return out


def _short(obj: Any) -> str:
    try:
        return json.dumps(obj, default=str)[:200]
    except Exception:
        return str(obj)[:200]
