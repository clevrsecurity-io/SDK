"""clevr_pydantic_ai — drop-in Pydantic AI integration for Clevr.

Pydantic AI (`pydantic-ai`) defines tools either with the `@agent.tool`
decorator or the `Tool(fn, ...)` constructor. Both ultimately end up as
callables on the Agent. We provide:

  • `guarded(clevr, ...)` — decorator that gates a tool function, equivalent
    in spirit to the LangChain decorator. Apply BEFORE `@agent.tool`.

  • `guard_agent(clevr, agent)` — monkey-patch every tool already attached
    to an Agent, gating its `function` with Clevr.

Pydantic AI tools receive a `RunContext` as their first argument when they
are dependency-injected, or just plain kwargs otherwise. The wrapper preserves
the original signature so Pydantic AI's schema introspection still works.

Usage:

    from pydantic_ai import Agent, RunContext
    from clevr import Clevr
    from clevr_pydantic_ai import guarded, guard_agent

    clevr = Clevr(agent="researcher")
    agent = Agent("openai:gpt-4o", system_prompt="You are a researcher.")

    @agent.tool
    @guarded(clevr, tool_name="kb_search", action_type="read")
    async def kb_search(ctx: RunContext, query: str) -> str:
        return await my_kb.search(query)

    # OR — gate every tool already attached to an agent:
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
    """Wrap a tool function with Clevr.guard. Sync and async-safe."""
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
                "action_type": action_type,
                "action": f"{name}({_short(kwargs)})",
                "target": target,
                "environment": environment,
                "metadata": {"args": _short(args), "kwargs": _short(kwargs)},
            }

        if is_async:
            @wraps(fn)
            async def awrapper(*args: Any, **kwargs: Any) -> Any:
                action = _action(args, kwargs)
                # The base Python SDK's guard() is sync — we run it in the
                # event loop's default executor to avoid blocking. Since the
                # underlying evaluate() is a single HTTP POST (~5ms in
                # enforce mode without LLM), this is fine.
                try:
                    return await _async_guard(clevr, action, fn, args, kwargs)
                except ClevrBlockedError as e:
                    return f"[Clevr blocked this action: {e.verdict.get('reason', '')}]"
                except ClevrEscalatedError as e:
                    return (
                        "[Clevr requires human approval before this action runs: "
                        f"{e.verdict.get('reason', '')}]"
                    )
            awrapper.__clevr_gated__ = True  # type: ignore[attr-defined]
            return awrapper

        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            action = _action(args, kwargs)
            try:
                return clevr.guard(action, run=lambda _v: fn(*args, **kwargs))
            except ClevrBlockedError as e:
                return f"[Clevr blocked this action: {e.verdict.get('reason', '')}]"
            except ClevrEscalatedError as e:
                return (
                    "[Clevr requires human approval before this action runs: "
                    f"{e.verdict.get('reason', '')}]"
                )
        wrapper.__clevr_gated__ = True  # type: ignore[attr-defined]
        return wrapper
    return decorator


def guard_agent(
    clevr: Clevr,
    agent: Any,
    *,
    meta_by_name: Optional[dict] = None,
) -> Any:
    """Gate every tool already attached to a Pydantic AI Agent in place.

    Returns the same agent for chaining. Works against `agent._function_tools`
    (Pydantic AI keeps a private dict of registered tools). If the internal
    name changes in a future release this will need an adjustment — we keep
    it isolated here on purpose.
    """
    meta_by_name = meta_by_name or {}
    # Pydantic AI <= 0.0.x stores tools under several internal names depending
    # on version; we probe in order. None is fatal — just no-op so callers
    # don't crash on a version skew.
    tools_map = (
        getattr(agent, "_function_tools", None)
        or getattr(agent, "_tools", None)
        or getattr(agent, "function_tools", None)
    )
    if not tools_map:
        return agent

    items = tools_map.items() if hasattr(tools_map, "items") else enumerate(tools_map)
    for key, tool in items:
        name = getattr(tool, "name", None) or (key if isinstance(key, str) else None)
        if not name:
            continue
        # Pydantic AI Tool objects expose `.function` (the callable). Some
        # versions also expose `.func`. Try both.
        func_attr = "function" if hasattr(tool, "function") else "func" if hasattr(tool, "func") else None
        if not func_attr:
            continue
        original_fn = getattr(tool, func_attr)
        if getattr(original_fn, "__clevr_gated__", False):
            continue  # already wrapped
        meta = meta_by_name.get(name, {})
        wrapped = guarded(clevr, tool_name=name, **meta)(original_fn)
        setattr(tool, func_attr, wrapped)
    return agent


async def _async_guard(
    clevr: Clevr,
    action: dict,
    fn: Callable,
    args: tuple,
    kwargs: dict,
) -> Any:
    """Async helper that mirrors clevr.guard() but awaits the inner coro."""
    verdict = clevr.evaluate(action)
    clevr.last_verdict = verdict

    if clevr.mode == "shadow":
        # Always run; surface lastVerdict for callers that want to log it.
        return await fn(*args, **kwargs)

    effect = verdict.get("effect")
    if effect == "allow":
        return await fn(*args, **kwargs)
    if effect == "block":
        raise ClevrBlockedError(verdict)
    # escalate
    if clevr.on_escalate == "allow":
        return await fn(*args, **kwargs)
    raise ClevrEscalatedError(verdict)


def _short(obj: Any) -> str:
    try:
        return json.dumps(obj, default=str)[:200]
    except Exception:
        return str(obj)[:200]
