"""clevr_autogen — AutoGen (Microsoft Research) + AG2 fork integration.

AutoGen agents call tools registered via `@agent.tool` (AutoGen ≥0.4) or
`@agent.register_for_llm` + `@agent.register_for_execution` (AutoGen 0.2).
Both end up storing a callable in the agent's function map.

We provide:
  • `guarded(clevr, ...)` — decorator wrapping a plain function. Apply
    BEFORE the AutoGen registration decorator.
  • `guard_agent(clevr, agent)` — gate every function already registered
    on an agent in place. Works on both AutoGen 0.2 (function_map) and
    0.4 (tools list).

Usage:
    from autogen import AssistantAgent
    from clevr import Clevr
    from clevr_autogen import guard_agent

    clevr = Clevr(agent="researcher")
    agent = AssistantAgent("researcher", llm_config={...})

    @agent.register_for_execution()
    def kb_search(query: str) -> str:
        return my_kb.search(query)

    guard_agent(clevr, agent)
"""

from __future__ import annotations

import inspect
import json
from functools import wraps
from typing import Any, Callable, Optional

from clevr import Clevr, ClevrBlockedError, ClevrEscalatedError


def guarded(
    clevr: Clevr,
    *,
    tool_name: Optional[str] = None,
    action_type: str = "tool_call",
    target_of: Optional[Callable[[dict], Optional[str]]] = None,
    environment: Optional[str] = None,
) -> Callable:
    def decorator(fn: Callable) -> Callable:
        name = tool_name or fn.__name__
        is_async = inspect.iscoroutinefunction(fn)

        def _action(args: tuple, kwargs: dict) -> dict:
            target = None
            if target_of:
                try:
                    target = target_of(kwargs)
                except Exception:
                    target = None
            return {
                "tool": name,
                "runtime": getattr(clevr, "runtime", None) or "autogen",   # the platform, unless the client names one
                "action_type": action_type,
                "action": f"{name}({_short(kwargs)})",
                "target": target,
                "environment": environment,
                "metadata": {"args": _short(args), "kwargs": _short(kwargs)},
            }

        if is_async:
            @wraps(fn)
            async def awrap(*args, **kwargs):
                try:
                    v = clevr.evaluate(_action(args, kwargs))
                    clevr.last_verdict = v
                    if clevr.mode == "shadow":
                        return await fn(*args, **kwargs)
                    if v.get("effect") == "allow":
                        return await fn(*args, **kwargs)
                    if v.get("effect") == "block":
                        raise ClevrBlockedError(v)
                    if clevr.on_escalate == "allow":
                        return await fn(*args, **kwargs)
                    raise ClevrEscalatedError(v)
                except ClevrBlockedError as e:
                    return f"[Clevr blocked this action: {e.verdict.get('reason', '')}]"
                except ClevrEscalatedError as e:
                    return f"[Clevr requires human approval: {e.verdict.get('reason', '')}]"
            awrap.__clevr_gated__ = True  # type: ignore[attr-defined]
            return awrap

        @wraps(fn)
        def wrap(*args, **kwargs):
            try:
                return clevr.guard(_action(args, kwargs),
                                   run=lambda _v: fn(*args, **kwargs))
            except ClevrBlockedError as e:
                return f"[Clevr blocked this action: {e.verdict.get('reason', '')}]"
            except ClevrEscalatedError as e:
                return f"[Clevr requires human approval: {e.verdict.get('reason', '')}]"
        wrap.__clevr_gated__ = True  # type: ignore[attr-defined]
        return wrap
    return decorator


def guard_agent(
    clevr: Clevr,
    agent: Any,
    *,
    meta_by_name: Optional[dict] = None,
) -> Any:
    """Gate every tool/function already registered on an AutoGen agent.

    Handles both AutoGen 0.2 (`agent.function_map`) and 0.4 (`agent._tools`
    or `agent.tools`). Returns the same agent for chaining.
    """
    meta_by_name = meta_by_name or {}
    # AutoGen 0.2 path
    fmap = getattr(agent, "function_map", None)
    if isinstance(fmap, dict):
        for name, fn in list(fmap.items()):
            if getattr(fn, "__clevr_gated__", False):
                continue
            meta = meta_by_name.get(name, {})
            fmap[name] = guarded(clevr, tool_name=name, **meta)(fn)
    # AutoGen 0.4 path — tools is a list of Tool objects with a `.func` attr
    tools = getattr(agent, "_tools", None) or getattr(agent, "tools", None)
    if isinstance(tools, list):
        for t in tools:
            func_attr = "func" if hasattr(t, "func") else "function" if hasattr(t, "function") else None
            if not func_attr:
                continue
            fn = getattr(t, func_attr)
            if getattr(fn, "__clevr_gated__", False):
                continue
            name = getattr(t, "name", None) or fn.__name__
            meta = meta_by_name.get(name, {})
            setattr(t, func_attr, guarded(clevr, tool_name=name, **meta)(fn))
    return agent


def _short(obj: Any) -> str:
    try:
        return json.dumps(obj, default=str)[:200]
    except Exception:
        return str(obj)[:200]
