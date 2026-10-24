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

       @tool
       @guarded(clevr, tool_name="kb_search", action_type="read")
       def kb_search(query: str) -> str:
           return my_kb.search(query)

2. **Callback handler** — for code paths where tools come from elsewhere
   (LangGraph, agent executors). A LangChain BaseCallbackHandler:

       from clevr_langchain import ClevrCallbackHandler

       handler = ClevrCallbackHandler(clevr)
       agent_executor.invoke({"input": ...}, {"callbacks": [handler]})

   Or, with no callbacks to pass, `install(clevr)` (what `clevr.init()` does)
   adds the handler to every run LangChain configures in this process.

The handler captures `on_tool_start`/`on_tool_end` events and calls evaluate
on each. Block/escalate verdicts surface back as tool errors that the LLM
can react to. In shadow mode, decisions are recorded and tools run, except
where the floor refuses or holds: that is applied in both modes.
"""

from __future__ import annotations

import ast
import json
from functools import wraps
from typing import Any, Callable, Optional

from clevr import Clevr, ClevrBlockedError, ClevrEscalatedError, floor_stands, tool_args

# A tool input longer than this is not parsed for its arguments.
_MAX_INPUT_READ = 100_000

try:
    # A real callback handler: LangChain reads its ignore_* flags and calls every
    # event on it. A class that only duck-typed the tool events made each tool
    # call fail with AttributeError, before Clevr was asked (measured 2026-10-14,
    # langchain-core 0.2.43 and 1.6.10). Without LangChain it is never called.
    from langchain_core.callbacks import BaseCallbackHandler as _CallbackBase
except ImportError:  # pragma: no cover
    _CallbackBase = object


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
                # Whole, for a rule on an argument (target.amount > 1000); the
                # strings below are only for display.
                "target_attr": tool_args(fn, args, kwargs),
                "metadata": {"args": _short(args), "kwargs": _short(kwargs)},
            }
            return clevr.guard(action, run=lambda _v: fn(*args, **kwargs))
        return wrapper
    return decorator


class ClevrCallbackHandler(_CallbackBase):
    """LangChain callback handler that gates every tool call.

    Use this when you can't easily decorate each tool — e.g. when LangChain's
    AgentExecutor pulls tools from a registry. Hook it as a callback and
    every tool invocation is guarded transparently.
    """

    # A verdict that stops the tool must reach the caller: LangChain logs a
    # handler's exception and runs the tool anyway unless the handler asks for it
    # to be raised. An engine that cannot be reached stops the tool too, as it
    # does through guarded(). Not run_inline: an async run waits for the handler
    # in a worker thread either way, and inline it held the event loop for the
    # engine's whole answer (4 parallel calls, 500 ms each: 2.0 s, not 0.5 s).
    raise_error = True

    def __init__(self, clevr: Clevr, action_type: str = "tool_call") -> None:
        self.clevr = clevr
        self.action_type = action_type
        self._verdicts: dict[str, dict] = {}  # run_id → verdict

    def on_tool_start(self, serialized: dict, input_str: str, *, run_id: Any, **kwargs: Any) -> None:
        tool_name = (serialized or {}).get("name") or "unknown_tool"
        action = {
            "tool": tool_name,
            "runtime": getattr(self.clevr, "runtime", None) or "langchain",
            "action_type": self.action_type,
            "action": f"{tool_name}({_truncate(input_str, 200)})",
            "target_attr": _input_args(input_str, kwargs.get("inputs")),
            "metadata": {"input": _truncate(input_str, 500)},
        }
        verdict = self.clevr.evaluate(action)
        self._verdicts[str(run_id)] = verdict
        if self.clevr.mode == "enforce" or floor_stands(verdict):
            effect = verdict.get("effect")
            # Refused here, before the tool: LangChain does not call on_tool_error
            # for what on_tool_start raises, so the report goes now.
            if effect == "block":
                self._refused(run_id, verdict)
                raise ClevrBlockedError(verdict)
            # 'step_up' is the legacy synonym for a held/escalate verdict; without
            # it a step_up tool call would run instead of being held.
            if effect in ("escalate", "step_up") and self.clevr.on_escalate == "throw":
                self._refused(run_id, verdict)
                raise ClevrEscalatedError(verdict)
            # Waits for a person, as guarded() does: approved, the approval is spent
            # by asking again with it and LangChain runs the tool; rejected, timed
            # out or already waiting (held_as), it raises and the tool does not run.
            if effect in ("escalate", "step_up") and self.clevr.on_escalate == "wait":
                try:
                    self._verdicts[str(run_id)] = self.clevr.wait_for_decision(verdict, action)
                except BaseException:
                    self._verdicts.pop(str(run_id), None)
                    raise
        # shadow mode: let the tool run, retain verdict for on_tool_end

    # The tool answered, or raised: the decision it ran on (the hold's approval
    # spent, when it waited) is told, as guarded() tells it.
    def on_tool_end(self, output: str, *, run_id: Any, **kwargs: Any) -> None:
        verdict = self._verdicts.pop(str(run_id), None)
        if verdict:
            self.clevr.report_enforcement(verdict.get("decision_id"), "allowed")
        if verdict and self.clevr.mode == "shadow" and verdict.get("effect") != "allow":
            # Caller might log this if they want; we keep it on the client
            # via clevr.last_verdict so the next code line can inspect.
            self.clevr.last_verdict = verdict

    def on_tool_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
        # A tool error does not establish that the operation had no side effects.
        self._verdicts.pop(str(run_id), None)

    def _refused(self, run_id: Any, verdict: dict) -> None:
        self._verdicts.pop(str(run_id), None)
        self.clevr.report_enforcement(verdict.get("decision_id"), "denied")


_INSTALLED: Optional[ClevrCallbackHandler] = None


def install(clevr: Clevr) -> ClevrCallbackHandler:
    """Gate every LangChain tool call in this process through `clevr`, with no
    callbacks to pass: the handler joins each run LangChain configures, the way
    its own tracer does. Called again, the gate moves to the new client."""
    global _INSTALLED
    if _INSTALLED is None:
        from contextvars import ContextVar
        from langchain_core.tracers.context import register_configure_hook
        _INSTALLED = ClevrCallbackHandler(clevr)
        # The variable's default, not a value set here: a thread started later
        # has none of this context and still sees it.
        register_configure_hook(ContextVar("clevr_callback_handler", default=_INSTALLED), True)
    else:
        _INSTALLED.clevr = clevr
    return _INSTALLED


def _input_args(input_str: Any, inputs: Any) -> Optional[dict]:
    """The tool's arguments from what on_tool_start receives. langchain-core
    passes the input dict as `inputs`; older releases pass only input_str, which
    is the JSON the model wrote, or str() of the dict, a Python literal."""
    if isinstance(inputs, dict):
        return tool_args(None, (), inputs)
    s = input_str.strip() if isinstance(input_str, str) else ""
    if not s.startswith("{") or len(s) > _MAX_INPUT_READ:
        return None
    try:
        parsed = json.loads(s)
    except ValueError:
        try:
            parsed = ast.literal_eval(s)
        except Exception:
            return None
    return tool_args(None, (), parsed) if isinstance(parsed, dict) else None


def _short(obj: Any) -> str:
    try:
        return json.dumps(obj, default=str)[:200]
    except Exception:
        return str(obj)[:200]


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"
