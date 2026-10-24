"""clevr — Python SDK for Clevr agent governance.

API parity with the Node @clevr/sdk:
    from clevr import Clevr, chain

    clevr = Clevr(api_key="clevr_sk_...", agent="oncall-agent", mode="shadow")
    clevr.with_root_human("alice@acme.com", display="Alice · On-call")

    verdict = clevr.guard(
        {"tool": "kb.search", "target": "kb/incidents", "action_type": "read",
         "action": "search for similar incidents"},
        run=lambda v: kb_client.search("incidents"),
    )

Pure standard library — no requests / httpx dependency, so it drops into
any Python project (LangChain, CrewAI, custom) without conflicts.

Verifiable identity (optional): pass ``identity_seed`` (a raw-32 Ed25519 seed,
base64) or set ``CLEVR_IDENTITY_SEED`` and every evaluate call is SIGNED, so the
engine proves this workload is who it claims (asserted -> verified). Provision
one with ``generate_identity_seed()``. This opt-in path needs the ``cryptography``
package; unsigned clients keep working as 'asserted' with pure stdlib.
"""

from __future__ import annotations

import dataclasses
import datetime
import enum
import inspect
import json
import math
import numbers
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any, Callable, Optional


# The policies the floor itself decides (brain lib/floor/floor.js), read when an
# engine older than `floor_enforced` answers.
_FLOOR_DECIDERS = {"content-risk", "verb-risk", "identity", "idp-deprovisioned", "session-stopped", "session-proof"}


def floor_stands(verdict: Optional[dict]) -> bool:
    """The floor's own refusal, or the hold that took its place: applied in every mode."""
    v = verdict or {}
    if v.get("floor_enforced") is True:
        return True
    return (v.get("effect") == "block" and v.get("matched_policy") in _FLOOR_DECIDERS) or (
        v.get("effect") == "escalate" and bool(v.get("floor_hold")))


class ClevrBlockedError(Exception):
    """Raised when the engine blocks an action in enforce mode."""

    def __init__(self, verdict: dict):
        super().__init__(verdict.get("reason") or "Action blocked by Clevr policy.")
        self.verdict = verdict


class ClevrEscalatedError(Exception):
    """Raised when the engine escalates an action and on_escalate='throw'."""

    def __init__(self, verdict: dict):
        super().__init__(verdict.get("reason") or "Action requires human approval.")
        self.verdict = verdict


class Clevr:
    """Single-tenant Clevr client.

    Args:
        api_key: Bearer key (`clevr_sk_...`). Falls back to env CLEVR_API_KEY.
        base:    Engine URL, default http://localhost:8787 or env CLEVR_URL.
        agent:   Agent name (used in audit + as last hop of synthesized chain).
        mode:    'enforce' (default — blocks/escalates per verdict) or 'shadow'
                 (records a discretionary verdict and lets the action run; the
                 floor's own refusal or hold is applied as in enforce).
        on_escalate: 'throw' (default) | 'wait' (polls for resolution) |
                     'allow' (runs anyway — dev only).
        session_id:   Optional session id sent on every call. Prefer start_session()
                      at the start of each run. Without one, the engine groups
                      the agent's calls by silence (a new session after 30 min).
        session_goal: Optional human-readable goal text (sealed on first decision).
        actor_chain:  Optional pre-built chain (root-first list of hops).
        timeout:      HTTP timeout in seconds (default 10).
        runtime:      The platform the agent runs on (langgraph, crewai, n8n...),
                      recorded on every decision so the console says where an
                      action came from. Env CLEVR_RUNTIME. Each adapter names
                      its own framework when this is unset. Recorded, never
                      judged: no verdict depends on it.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base: Optional[str] = None,
        agent: str = "unnamed-agent",
        mode: str = "enforce",
        on_escalate: str = "throw",
        session_id: Optional[str] = None,
        session_goal: Optional[str] = None,
        actor_chain: Optional[list[dict]] = None,
        timeout: float = 10.0,
        identity_seed: Optional[str] = None,
        runtime: Optional[str] = None,
    ) -> None:
        self.base = (base or os.environ.get("CLEVR_URL") or "http://localhost:8787").rstrip("/")
        self.api_key = api_key or os.environ.get("CLEVR_API_KEY")
        if not self.api_key:
            raise ValueError("Clevr SDK: api_key is required (or set CLEVR_API_KEY).")
        self.agent = agent
        self.runtime = runtime or os.environ.get("CLEVR_RUNTIME") or None
        if mode not in ("enforce", "shadow"):
            raise ValueError(f"Clevr SDK: mode must be 'enforce' or 'shadow' (got {mode!r})")
        self.mode = mode
        if on_escalate not in ("throw", "wait", "allow"):
            raise ValueError(f"Clevr SDK: on_escalate must be 'throw'|'wait'|'allow'")
        self.on_escalate = on_escalate
        self.session_id = session_id
        self.session_goal = session_goal
        self.actor_chain = list(actor_chain) if actor_chain else None
        self.timeout = timeout
        # Optional verifiable identity: a raw-32 Ed25519 seed (base64). When set,
        # every evaluate call is SIGNED so the engine verifies WHO is acting
        # (asserted -> verified). Provision one with generate_identity_seed().
        # Needs the optional 'cryptography' package; unsigned clients keep working.
        self.identity_seed = identity_seed or os.environ.get("CLEVR_IDENTITY_SEED")
        self.last_verdict: Optional[dict] = None
        self.last_request_id: Optional[str] = None
        self._reports: list[threading.Thread] = []

    # ─── Chain helpers ───────────────────────────────────────────────────

    def start_session(self, goal: Optional[str] = None) -> str:
        """Start a new session on this client: one per run of the agent. Every
        later call carries it until the next start_session. Returns the id."""
        self.session_id = "sess_" + uuid.uuid4().hex[:10]
        if goal is not None:
            self.session_goal = goal
        return self.session_id

    def with_root_human(self, human_id: str, display: Optional[str] = None) -> list[dict]:
        """Initialize the actor chain with a human → agent pair. Returns the chain."""
        human = {"type": "human", "id": human_id}
        if display:
            human["display"] = display
        self.actor_chain = [
            human,
            {"type": "agent", "id": self.agent, "on_behalf_of": human_id},
        ]
        return self.actor_chain

    def child(self, child_agent_id: str) -> "Clevr":
        """Return a new Clevr scoped to a sub-agent — inherits session + chain."""
        parent_chain = self.actor_chain or [{"type": "agent", "id": self.agent}]
        last = parent_chain[-1]
        new_chain = list(parent_chain) + [
            {"type": "agent", "id": child_agent_id, "on_behalf_of": last.get("id", self.agent)},
        ]
        return Clevr(
            api_key=self.api_key, base=self.base, agent=child_agent_id,
            mode=self.mode, on_escalate=self.on_escalate,
            session_id=self.session_id, session_goal=self.session_goal,
            actor_chain=new_chain, timeout=self.timeout,
            identity_seed=self.identity_seed, runtime=self.runtime,
        )

    # ─── Core verbs ──────────────────────────────────────────────────────

    def evaluate(self, action: dict) -> dict:
        """Call /v1/evaluate without running anything. Returns the verdict dict."""
        body: dict[str, Any] = {
            "agent": self.agent,
            "session_id": self.session_id,
            "session_goal": self.session_goal,
            # The action's own runtime, when it names one, wins (it comes after).
            **({"runtime": self.runtime} if self.runtime else {}),
            **action,
        }
        if self.actor_chain:
            body["actor_chain"] = self.actor_chain
        # Surface tool-call arguments as target_attr so the engine's deterministic
        # Layer-4 parameter rules (target.<field>, e.g. amount > 1000) can read
        # them. Adapters stash the raw args under metadata.input / metadata.args;
        # promote them unless the caller set target_attr explicitly.
        if body.get("target_attr") is None and isinstance(body.get("metadata"), dict):
            a = body["metadata"].get("input")
            if a is None:
                a = body["metadata"].get("args")
            if isinstance(a, dict):
                body["target_attr"] = a
        # Sign the request with this workload's identity key (if provisioned) so
        # the engine verifies WHO is acting. Bound to the exact agent + action so
        # a captured proof can't be lifted onto a different agent or tool. Never
        # let signing break the call — on any error the identity stays 'asserted'.
        if self.identity_seed:
            try:
                from clevr_identity import sign_identity_proof
                body["identity"] = sign_identity_proof(self.identity_seed, {
                    "agent": body.get("agent"),
                    "action_type": body.get("action_type"),
                    "action": body.get("action"),
                    "tool": body.get("tool"),
                    "session_id": body.get("session_id"),
                })
            except Exception:
                pass
        verdict = self._post("/v1/evaluate", body)
        # An unknown response is not an escalation that may be auto-allowed.
        if not isinstance(verdict, dict) or verdict.get("effect") not in ("allow", "block", "escalate", "step_up"):
            raise ValueError("Clevr evaluate returned an invalid verdict")
        self.last_verdict = verdict
        # A session the engine opened is not taken over: kept on a long-lived
        # client, it would make every later run one endless session.
        if verdict.get("request_id"):
            self.last_request_id = verdict["request_id"]
        return verdict

    def request_card_payment(self, amount, currency: str, merchant: Optional[str] = None,
                             reason: Optional[str] = None, resume: Optional[str] = None) -> dict:
        """Ask to pay with this agent's card (POST /v1/card/spend).

        Judged like any action. On an allow Clevr opens a one-time permit; the
        next purchase on the card that fits it is approved when the card network
        asks. The card number is never part of this exchange. Returns the answer
        with `status`: 200 granted, 202 held (ask again with resume=decision_id
        once a person approved), 403 refused, 409 allowed but no permit opened.
        """
        body = {"agent_id": self.agent, "session_id": self.session_id,
                "amount": str(amount), "currency": currency, "merchant": merchant, "reason": reason}
        if resume:
            body["resume"] = resume
        req = urllib.request.Request(
            f"{self.base}/v1/card/spend", data=json.dumps(body).encode("utf-8"), method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return {"status": r.status, **json.loads(r.read().decode("utf-8"))}
        except urllib.error.HTTPError as e:
            text = e.read().decode("utf-8", "ignore")
            if e.code in (202, 403, 409):
                try:
                    return {"status": e.code, **json.loads(text)}
                except ValueError:
                    pass
            raise RuntimeError(f"Clevr /v1/card/spend {e.code}: {text[:200]}")

    def guard(self, action: dict, run: Callable[[dict], Any]) -> Any:
        """Wrap a tool execution with the policy gate.

        In `enforce` mode: returns the inner's result on allow, raises
        ClevrBlockedError on block, behaves per on_escalate on escalate.
        In `shadow` mode: runs the inner and returns a dict
        {result, verdict, shadow, would_have_blocked, would_have_escalated},
        whatever a discretionary verdict says. The floor's refusal or hold is
        applied as in enforce: a refusal raises, a hold follows on_escalate.
        """
        verdict = self.evaluate(action)
        if self.mode != "shadow":
            return self._apply(verdict, run, action)
        return {
            "result": self._apply(verdict, run, action) if floor_stands(verdict) else self.run_reported(verdict, run),
            "verdict": verdict,
            "shadow": True,
            "would_have_blocked": verdict.get("effect") == "block",
            "would_have_escalated": verdict.get("effect") == "escalate",
        }

    def _apply(self, verdict: dict, run: Callable[[dict], Any], action: dict) -> Any:
        effect = verdict.get("effect")
        if effect == "allow":
            return self.run_reported(verdict, run)
        if effect == "block":
            self.report_enforcement(verdict.get("decision_id"), "denied")
            raise ClevrBlockedError(verdict)
        # escalate
        if self.on_escalate == "allow":
            return self.run_reported(verdict, run)
        if self.on_escalate == "wait":
            return self._wait_then_run(verdict, run, action)
        self.report_enforcement(verdict.get("decision_id"), "denied")
        raise ClevrEscalatedError(verdict)

    # ─── What this door did with the verdict ─────────────────────────────
    # Only a pre-execution refusal proves the gate prevented the call.
    # A tool exception leaves side effects unconfirmed, even after an approval.

    def report_enforcement(self, decision_id: Optional[str], enforced: str) -> None:
        """Tell Clevr what this door did with a decision: 'allowed', 'denied'
        or 'asked'. Sent on a thread of its own, never raises, never waits:
        a report that cannot be sent leaves the decision unconfirmed."""
        if not decision_id or enforced not in ("allowed", "asked", "denied"):
            return
        t = threading.Thread(target=self._send_report, args=(decision_id, enforced), name="clevr-report")
        t.start()
        if len(self._reports) > 64:
            self._reports = [x for x in self._reports if x.is_alive()]
        self._reports.append(t)

    def flush(self, timeout: float = 5.0) -> None:
        """Wait for the reports still on their way, up to `timeout` seconds
        in all. Call it before a short-lived process exits."""
        deadline = time.time() + timeout
        for t in self._reports:
            t.join(max(0.0, deadline - time.time()))
        self._reports = [x for x in self._reports if x.is_alive()]

    def _send_report(self, decision_id: str, enforced: str) -> None:
        req = urllib.request.Request(
            f"{self.base}/v1/decisions/{urllib.parse.quote(decision_id, safe='')}/enforcement",
            data=json.dumps({"enforced": enforced}).encode("utf-8"), method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=min(self.timeout, 2.0)) as r:
                r.read()
        except Exception:
            pass

    def run_reported(self, verdict: dict, run: Callable[[dict], Any]) -> Any:
        """Run the tool on a verdict and report how it went."""
        # An exception cannot prove that no side effect occurred.
        result = run(verdict)
        self.report_enforcement(verdict.get("decision_id"), "allowed")
        return result

    async def arun_reported(self, verdict: dict, run: Callable[[], Any]) -> Any:
        """run_reported for a coroutine: `run()` returns the awaitable."""
        # Cancellation or failure after launch leaves execution uncertain.
        result = await run()
        self.report_enforcement(verdict.get("decision_id"), "allowed")
        return result

    # ─── Internal ────────────────────────────────────────────────────────

    def _post(self, path: str, body: dict) -> dict:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base}{path}",
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Clevr {path} {e.code}: {e.read().decode('utf-8', 'ignore')[:200]}")

    def _get(self, path: str) -> dict:
        req = urllib.request.Request(
            f"{self.base}{path}",
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def _wait_then_run(self, verdict: dict, run: Callable, action: dict, timeout_s: float = 600.0) -> Any:
        """Wait for a person, then run the tool on the approval and report how it went."""
        return self.run_reported(self.wait_for_decision(verdict, action, timeout_s), run)

    def wait_for_decision(self, verdict: dict, action: dict, timeout_s: float = 600.0) -> dict:
        """Poll the decision endpoint until the escalation is resolved by a human.
        Waits until the decision's deadline when it carries one, else timeout_s.
        Returns the verdict that spent the approval, for the caller to run on;
        raises ClevrBlockedError on a rejection and ClevrEscalatedError on a
        timeout or a hold the same action already waits on (held_as)."""
        # The same action already waits for a person (held_as): this attempt is
        # settled with that request and never runs on its own, so it does not wait.
        if verdict.get("held_as"):
            self.report_enforcement(verdict.get("decision_id"), "denied")
            raise ClevrEscalatedError(verdict)
        decision_id = verdict.get("decision_id")
        start = time.time()
        deadline = start + timeout_s
        delay = 1.0
        while time.time() < deadline:
            time.sleep(delay)
            delay = min(delay * 1.5, 15.0)
            try:
                status = self._get(f"/v1/decisions/{decision_id}") or {}
            except Exception:
                continue
            # The wait ends when the decision ends, as in the JS SDK: the workspace
            # sets how long a held action may wait before it is declined on its own
            # and the decision carries that deadline. A fixed 600 s had a workspace
            # allowing thirty minutes see its Python agents give up at ten. A minute
            # and five seconds past it, because the sweep that declines it runs once
            # a minute: the agent reads that answer instead of a timeout.
            ends = _timestamp(status.get("expires_at"))
            if ends is not None:
                deadline = ends + 65.0
            # 'allow' / 'block' are the raw override effects, read like the
            # resolutions they stand for, so a direct effect write still ends the wait.
            resolution = status.get("resolution")
            if resolution in ("approved", "allow"):
                # Run on the approval, spent by asking again with it, as every
                # other door does: run on the poll alone it was never spent.
                spent = self.evaluate({**action, "resume": decision_id})
                if spent.get("effect") == "allow":
                    return spent
                self.report_enforcement(spent.get("decision_id"), "denied")
                if spent.get("effect") == "block":
                    raise ClevrBlockedError(spent)
                raise ClevrEscalatedError(spent)
            if resolution in ("rejected", "block"):
                self.report_enforcement(decision_id, "denied")
                raise ClevrBlockedError({
                    **verdict,
                    "reason": f"Rejected by {status.get('resolved_by')}: {status.get('resolution_reason')}",
                })
        self.report_enforcement(decision_id, "denied")
        raise ClevrEscalatedError({**verdict, "reason": f"Escalation timed out after {round(time.time() - start)}s."})


def _timestamp(value: Any) -> Optional[float]:
    """An ISO 8601 date as epoch seconds, or None when there is none to read."""
    if not isinstance(value, str) or not value:
        return None
    # fromisoformat reads a trailing 'Z' only from Python 3.11.
    text = value[:-1] + "+00:00" if value.endswith(("Z", "z")) else value
    try:
        return datetime.datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


# ── The arguments of a tool call, for target_attr ───────────────────
# A rule on an argument (target.amount > 1000) reads target_attr, and until
# 2026-10-14 no adapter filled it: each sent its arguments as a 200-character
# JSON string under metadata, which evaluate() does not promote, so on every
# Python framework door such a rule never fired.

_SKIPPED = object()
# Deep enough for any real argument, well inside the engine's 64-level cap on a
# body (a deeper one is refused with a 400 and the tool would fail).
_MAX_ARG_DEPTH = 24
# What a framework hands a tool alongside the model's arguments: the run itself,
# not the call. A Kernel would dump every plugin and service it holds.
_INJECTED = frozenset({"RunContext", "Kernel"})


def tool_args(fn: Any, args: tuple = (), kwargs: Optional[dict] = None) -> Optional[dict]:
    """The arguments of one tool call by parameter name, as JSON values.

    Positional arguments are named from fn's signature, so pay("acme", 5000)
    reads like pay(vendor="acme", amount=5000). A value no rule can read (an
    object, `self`, the framework's run context) is left out. None when no
    argument is left.
    """
    named: dict = dict(kwargs or {})
    if args:
        named["_args"] = list(args)
    if fn is not None:
        try:
            bound = inspect.signature(fn).bind_partial(*args, **(kwargs or {}))
        except (TypeError, ValueError):
            bound = None
        if bound is not None:
            named = {}
            for name, value in bound.arguments.items():
                kind = bound.signature.parameters[name].kind
                if kind is inspect.Parameter.VAR_KEYWORD:
                    named.update(value)
                elif name not in ("self", "cls"):
                    named[name] = list(value) if kind is inspect.Parameter.VAR_POSITIONAL else value
    out = _plain(named, 0)
    return out or None


def _plain(v: Any, depth: int) -> Any:
    if v is None or isinstance(v, (bool, str)):
        return v
    if isinstance(v, int):
        return int(v)
    if isinstance(v, numbers.Number):
        # Decimal, numpy scalars. NaN and infinity are not JSON (the engine would
        # refuse the body): sent as words, an amount rule reads them as unreadable
        # and a hold still applies.
        try:
            f = float(v)
        except (TypeError, ValueError):
            return str(v)
        return f if math.isfinite(f) else str(v)
    if isinstance(v, enum.Enum):
        return _plain(v.value, depth)
    if isinstance(v, (datetime.date, datetime.time, uuid.UUID)):
        return v.isoformat() if hasattr(v, "isoformat") else str(v)
    if depth >= _MAX_ARG_DEPTH or type(v).__name__ in _INJECTED:
        return _SKIPPED
    if hasattr(v, "model_dump") and callable(v.model_dump):
        try:
            return _plain(v.model_dump(mode="json"), depth + 1)
        except Exception:
            return _SKIPPED
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return _plain({f.name: getattr(v, f.name, None) for f in dataclasses.fields(v)}, depth + 1)
    if isinstance(v, dict):
        out = {}
        for k, x in v.items():
            p = _plain(x, depth + 1)
            if p is not _SKIPPED:
                out[str(k)] = p
        return out
    if isinstance(v, (list, tuple, set, frozenset)):
        return [p for p in (_plain(x, depth + 1) for x in v) if p is not _SKIPPED]
    return _SKIPPED


# ── Auto-instrumentation (two-line auto-install) ─────────────────
# After `clevr.init(...)`, every LangChain / CrewAI / Pydantic AI tool that
# gets created (or already exists) is automatically gated by the singleton
# returned. The user never has to wrap a `@guarded(...)` or `guard(...)` call
# manually — that's still available for advanced cases, but the floor for
# "Clevr in my agent" is now 2 lines:
#
#     import clevr
#     clevr.init(api_key="clevr_sk_…", agent="oncall", mode="shadow")
#
# Frameworks are detected lazily: we patch only the ones that are already
# imported. Frameworks imported AFTER init() are not auto-instrumented —
# this is a deliberate constraint (no global import hook magic) that keeps
# the behaviour predictable and audit-friendly. Call `clevr.instrument(...)`
# again after a late import if you need it.

_SINGLETON: Optional["Clevr"] = None


def init(
    api_key: Optional[str] = None,
    base: Optional[str] = None,
    agent: str = "unnamed-agent",
    mode: str = "enforce",
    on_escalate: str = "throw",
    auto_instrument: bool = True,
    frameworks: Optional[list[str]] = None,
) -> "Clevr":
    """Initialize Clevr and auto-instrument any imported framework.

    Args:
        api_key, base, agent, mode, on_escalate: same as the Clevr constructor.
        auto_instrument: if True (default), patch detected frameworks in-place.
        frameworks: optional explicit list — one of {"langchain", "crewai",
            "pydantic_ai"}. If None, every supported framework that's already
            imported is instrumented.

    Returns the Clevr singleton. The same instance is also stored module-wide
    so `clevr.singleton()` works from anywhere.
    """
    global _SINGLETON
    _SINGLETON = Clevr(
        api_key=api_key, base=base, agent=agent,
        mode=mode, on_escalate=on_escalate,
    )
    if auto_instrument:
        instrument(_SINGLETON, frameworks=frameworks)
    return _SINGLETON


def singleton() -> Optional["Clevr"]:
    """Return the Clevr instance created by init(), or None if init() was not called."""
    return _SINGLETON


def instrument(
    clevr_instance: "Clevr",
    frameworks: Optional[list[str]] = None,
) -> dict[str, bool]:
    """Patch detected frameworks to route through clevr_instance.

    Returns a dict {framework: was_patched}. Frameworks not imported are
    skipped silently (was_patched=False). We don't raise on missing
    framework — the design is opportunistic.
    """
    import sys
    wanted = set(frameworks) if frameworks else None
    result: dict[str, bool] = {"langchain": False, "crewai": False, "pydantic_ai": False}

    # ── LangChain ────────────────────────────────────────────────────────
    if (wanted is None or "langchain" in wanted) and \
       ("langchain_core" in sys.modules or "langchain" in sys.modules):
        # Through LangChain's own configure hook. It was looked for where it is
        # not, and the fallback set an attribute LangChain never reads: init()
        # answered langchain=True and a 5000 payment ran without Clevr being
        # asked (measured 2026-10-14, langchain-core 0.2.43 and 1.6.10).
        try:
            from clevr_langchain import install as _install_langchain  # type: ignore
            _install_langchain(clevr_instance)
            result["langchain"] = True
        except Exception:
            pass

    # ── CrewAI ───────────────────────────────────────────────────────────
    if (wanted is None or "crewai" in wanted) and "crewai" in sys.modules:
        try:
            from clevr_crewai import guard as _guard  # type: ignore
            from crewai.tools import BaseTool  # type: ignore
            # Monkey-patch __init_subclass__ so any *future* tool defined
            # after init() is also gated. For already-existing instances,
            # we walk Crew.agents and patch their .tools.
            original_init_subclass = BaseTool.__init_subclass__
            def _gated_subclass(cls, **kw):  # noqa: ANN001
                original_init_subclass(**kw) if original_init_subclass else None
                original_init = cls.__init__
                def _gated_init(self, *a, **kw):
                    original_init(self, *a, **kw)
                    _guard(clevr_instance, self)
                cls.__init__ = _gated_init
            BaseTool.__init_subclass__ = classmethod(_gated_subclass)
            result["crewai"] = True
        except Exception:
            pass

    # ── Pydantic AI ──────────────────────────────────────────────────────
    if (wanted is None or "pydantic_ai" in wanted) and "pydantic_ai" in sys.modules:
        try:
            from clevr_pydantic_ai import guard_agent as _guard_agent  # type: ignore
            from pydantic_ai import Agent  # type: ignore
            # Monkey-patch Agent so any agent instantiated post-init() is
            # auto-gated. Existing agents would need a manual guard_agent().
            original_init = Agent.__init__
            def _gated_init(self, *a, **kw):
                original_init(self, *a, **kw)
                _guard_agent(clevr_instance, self)
            Agent.__init__ = _gated_init
            result["pydantic_ai"] = True
        except Exception:
            pass

    return result


def shutdown() -> None:
    """Best-effort cleanup. Today this clears the singleton; future versions
    may flush a pending decision queue or close streaming connections."""
    global _SINGLETON
    _SINGLETON = None


# Verifiable-identity helpers, re-exported so callers can `from clevr import
# generate_identity_seed`. clevr_identity is pure-stdlib at import time (it loads
# 'cryptography' lazily, only when actually signing), so this never breaks the
# unsigned pure-stdlib path.
try:
    from clevr_identity import generate_identity_seed, sign_identity_proof  # noqa: F401
except Exception:  # pragma: no cover - defensive; clevr_identity is stdlib-only at import
    pass


class chain:  # noqa: N801 — namespacing helper, lowercase by intent.
    """Module-level chain builders, mirroring the Node SDK's `chain.*`."""

    @staticmethod
    def from_human(human_id: str, agent_id: str, display: Optional[str] = None) -> list[dict]:
        """Build a 2-element chain [human, agent]."""
        human = {"type": "human", "id": human_id}
        if display:
            human["display"] = display
        return [human, {"type": "agent", "id": agent_id, "on_behalf_of": human_id}]

    @staticmethod
    def child(parent_chain: list[dict], agent_id: str, hop_type: str = "agent") -> list[dict]:
        last = parent_chain[-1] if parent_chain else None
        return list(parent_chain) + [
            {"type": hop_type, "id": agent_id, "on_behalf_of": (last or {}).get("id")},
        ]
