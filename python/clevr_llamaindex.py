"""clevr_llamaindex — LlamaIndex tool integration.

LlamaIndex defines tools as `FunctionTool` instances with a `.fn` callable.
Tools attach to ReActAgent / OpenAIAgent / FunctionCallingAgentWorker via
the `tools` arg at agent construction.

Two integration styles:

  • `guard(clevr, tool)` — wrap a FunctionTool in-place; mutates .fn so
    every call to the tool is gated by Clevr.
  • `guard_all(clevr, tools)` — bulk version for a list of tools, with
    optional per-tool metadata.

Usage:
    from llama_index.core.tools import FunctionTool
    from llama_index.core.agent import ReActAgent
    from clevr import Clevr
    from clevr_llamaindex import guard

    clevr = Clevr(agent="researcher")

    def kb_search(query: str) -> str:
        return my_kb.search(query)

    tool = guard(clevr, FunctionTool.from_defaults(
        fn=kb_search, name="kb_search",
        description="Search the knowledge base"
    ), action_type="read")

    agent = ReActAgent.from_tools([tool], llm=...)
"""

from __future__ import annotations

import json
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
    """Wrap a LlamaIndex FunctionTool so its `.fn` is gated by Clevr.

    Returns the same tool for chaining. Idempotent — re-wrapping a tool
    that's already gated is a no-op.
    """
    name = tool_name or getattr(tool, "metadata", None) and getattr(tool.metadata, "name", None) \
           or getattr(tool, "name", None) or "unknown_tool"

    # FunctionTool stores the callable on `.fn` (LlamaIndex 0.10+) or `._fn`.
    func_attr = "fn" if hasattr(tool, "fn") else "_fn" if hasattr(tool, "_fn") else None
    if not func_attr:
        return tool
    original = getattr(tool, func_attr)
    if getattr(original, "__clevr_gated__", False):
        return tool

    def gated(*args, **kwargs):
        target = None
        if target_of:
            try:
                target = target_of(kwargs)
            except Exception:
                target = None
        action = {
            "tool": name,
            "runtime": getattr(clevr, "runtime", None) or "llamaindex",   # the platform, unless the client names one
            "action_type": action_type,
            "action": f"{name}({_short(kwargs)})",
            "target": target,
            "environment": environment,
            "metadata": {"args": _short(args), "kwargs": _short(kwargs)},
        }
        try:
            return clevr.guard(action, run=lambda _v: original(*args, **kwargs))
        except ClevrBlockedError as e:
            return f"[Clevr blocked this action: {e.verdict.get('reason', '')}]"
        except ClevrEscalatedError as e:
            return f"[Clevr requires human approval: {e.verdict.get('reason', '')}]"
    gated.__clevr_gated__ = True  # type: ignore[attr-defined]
    setattr(tool, func_attr, gated)
    return tool


def guard_all(
    clevr: Clevr,
    tools: list,
    *,
    meta_by_name: Optional[dict] = None,
) -> list:
    meta_by_name = meta_by_name or {}
    out = []
    for t in tools:
        name = getattr(getattr(t, "metadata", None), "name", None) or getattr(t, "name", "")
        meta = meta_by_name.get(name, {})
        out.append(guard(clevr, t, **meta))
    return out


def _short(obj: Any) -> str:
    try:
        return json.dumps(obj, default=str)[:200]
    except Exception:
        return str(obj)[:200]
