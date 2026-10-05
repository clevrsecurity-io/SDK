"""clevr_bedrock — AWS Bedrock Agents tool integration.

Bedrock Agents call your tools via Action Groups served by Lambda or an
HTTP webhook. The agent's `bedrock-agent-runtime:InvokeAgent` API passes
each tool call as an `actionGroupInvocation` event.

We provide:
  • `wrap_lambda_handler(handler, clevr)` — drop-in around any Lambda
    handler that processes Bedrock Action Group events. Each tool call
    is gated before your handler runs.
  • `guard(clevr, fn, ...)` — generic decorator for any callable that
    receives `(action_group, function, parameters)` tuples directly.

Usage in a Lambda action handler:

    from clevr import Clevr
    from clevr_bedrock import wrap_lambda_handler

    clevr = Clevr(agent="bedrock-agent")

    def original_handler(event, context):
        ag = event["actionGroup"]
        fn = event["function"]
        params = {p["name"]: p["value"] for p in event.get("parameters", [])}
        # ... your real logic ...
        return {"response": {"body": "result"}}

    lambda_handler = wrap_lambda_handler(original_handler, clevr)
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

from clevr import Clevr, ClevrBlockedError, ClevrEscalatedError


def wrap_lambda_handler(
    handler: Callable[[dict, Any], dict],
    clevr: Clevr,
    *,
    action_type: str = "tool_call",
) -> Callable[[dict, Any], dict]:
    """Wrap a Bedrock Agents Lambda handler with Clevr gating.

    Each event is inspected for actionGroup + function + parameters; we
    build a Clevr action, call evaluate, and either let the handler run
    or return an error payload Bedrock will surface to the model.
    """
    def gated_handler(event: dict, context: Any) -> dict:
        action_group = event.get("actionGroup", "unknown")
        fn_name = event.get("function", "unknown")
        params = {p.get("name"): p.get("value") for p in event.get("parameters", []) if isinstance(p, dict)}
        # Bedrock sessionId surfaces as 'sessionId' on the event.
        session_id = event.get("sessionId")
        if session_id and not clevr.session_id:
            clevr.session_id = session_id

        action = {
            "tool": f"{action_group}.{fn_name}",
            "runtime": getattr(clevr, "runtime", None) or "bedrock",   # the platform, unless the client names one
            "action_type": action_type,
            "action": f"{fn_name}({_short(params)})",
            "target": params.get("target") or params.get("path") or params.get("url"),
            "metadata": {"action_group": action_group, "params": params},
        }
        try:
            return clevr.guard(action, run=lambda _v: handler(event, context))
        except ClevrBlockedError as e:
            return _bedrock_error_response(action_group, fn_name,
                f"Clevr blocked this action: {e.verdict.get('reason', '')}")
        except ClevrEscalatedError as e:
            return _bedrock_error_response(action_group, fn_name,
                f"Clevr requires human approval: {e.verdict.get('reason', '')}")

    return gated_handler


def guard(
    clevr: Clevr,
    fn: Callable,
    *,
    tool_name: Optional[str] = None,
    action_type: str = "tool_call",
) -> Callable:
    """Wrap a plain action-group callable. The callable receives kwargs
    matched against the Bedrock function-schema parameters."""
    name = tool_name or getattr(fn, "__name__", "unknown_tool")

    def gated(**kwargs: Any) -> Any:
        action = {
            "tool": name,
            "runtime": getattr(clevr, "runtime", None) or "bedrock",
            "action_type": action_type,
            "action": f"{name}({_short(kwargs)})",
            "metadata": {"kwargs": _short(kwargs)},
        }
        try:
            return clevr.guard(action, run=lambda _v: fn(**kwargs))
        except ClevrBlockedError as e:
            return f"[Clevr blocked this action: {e.verdict.get('reason', '')}]"
        except ClevrEscalatedError as e:
            return f"[Clevr requires human approval: {e.verdict.get('reason', '')}]"
    gated.__clevr_gated__ = True  # type: ignore[attr-defined]
    return gated


def _bedrock_error_response(action_group: str, function: str, message: str) -> dict:
    """Format the Bedrock response so the agent's LLM reads it as a tool
    error and explains the situation to the user."""
    return {
        "messageVersion": "1.0",
        "response": {
            "actionGroup": action_group,
            "function": function,
            "functionResponse": {
                "responseBody": {"TEXT": {"body": message}},
                "responseState": "FAILURE",
            },
        },
    }


def _short(obj: Any) -> str:
    try:
        return json.dumps(obj, default=str)[:200]
    except Exception:
        return str(obj)[:200]
