"""clevr_semantic_kernel — Microsoft Semantic Kernel integration.

Semantic Kernel exposes tools as `KernelFunction` instances grouped in
`KernelPlugin`s. They're invoked by the kernel during function-calling
loops via `kernel.invoke()` or `kernel.invoke_function_call()`.

We provide:
  • `@guarded(clevr, ...)` — decorator wrapping a Python function
    BEFORE `@kernel_function` so SK still sees the right signature.
  • `guard_plugin(clevr, plugin)` — gate every KernelFunction in an
    existing KernelPlugin in place.

Usage:
    from semantic_kernel.functions import kernel_function
    from clevr import Clevr
    from clevr_semantic_kernel import guarded

    clevr = Clevr(agent="researcher")

    class KbPlugin:
        @kernel_function(description="Search the knowledge base.")
        @guarded(clevr, tool_name="kb_search", action_type="read")
        def kb_search(self, query: str) -> str:
            return my_kb.search(query)

    kernel.add_plugin(KbPlugin(), plugin_name="kb")
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
                "action_type": action_type,
                "action": f"{name}({_short(kwargs)})",
                "target": target,
                "environment": environment,
                "metadata": {"args": _short(args), "kwargs": _short(kwargs)},
            }

        if is_async:
            @wraps(fn)
            async def awrap(*args, **kwargs):
                v = clevr.evaluate(_action(args, kwargs))
                clevr.last_verdict = v
                try:
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


def guard_plugin(
    clevr: Clevr,
    plugin: Any,
    *,
    meta_by_name: Optional[dict] = None,
) -> Any:
    """Gate every KernelFunction in a KernelPlugin in place.

    SK stores functions on the plugin as `plugin.functions` (dict-like in
    modern SK). We replace each function's callable with the gated wrapper.
    """
    meta_by_name = meta_by_name or {}
    funcs = getattr(plugin, "functions", None) or getattr(plugin, "_functions", None)
    if not funcs:
        return plugin
    items = funcs.items() if hasattr(funcs, "items") else enumerate(funcs)
    for key, kf in items:
        name = getattr(kf, "name", None) or (key if isinstance(key, str) else None)
        if not name:
            continue
        func_attr = next((a for a in ("method", "function", "_function") if hasattr(kf, a)), None)
        if not func_attr:
            continue
        original = getattr(kf, func_attr)
        if getattr(original, "__clevr_gated__", False):
            continue
        meta = meta_by_name.get(name, {})
        setattr(kf, func_attr, guarded(clevr, tool_name=name, **meta)(original))
    return plugin


def _short(obj: Any) -> str:
    try:
        return json.dumps(obj, default=str)[:200]
    except Exception:
        return str(obj)[:200]
