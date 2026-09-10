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

import json
import os
import time
import urllib.error
import urllib.request
import uuid
from typing import Any, Callable, Optional


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
                 (records the verdict but always lets the action run).
        on_escalate: 'throw' (default) | 'wait' (polls for resolution) |
                     'allow' (runs anyway — dev only).
        session_id:   Optional explicit session id (else auto-generated).
        session_goal: Optional human-readable goal text (sealed on first decision).
        actor_chain:  Optional pre-built chain (root-first list of hops).
        timeout:      HTTP timeout in seconds (default 10).
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
    ) -> None:
        self.base = (base or os.environ.get("CLEVR_URL") or "http://localhost:8787").rstrip("/")
        self.api_key = api_key or os.environ.get("CLEVR_API_KEY")
        if not self.api_key:
            raise ValueError("Clevr SDK: api_key is required (or set CLEVR_API_KEY).")
        self.agent = agent
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

    # ─── Chain helpers ───────────────────────────────────────────────────

    def start_session(self, goal: Optional[str] = None) -> str:
        """Mint a new session_id. Returns it so the caller can persist it."""
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
            identity_seed=self.identity_seed,
        )

    # ─── Core verbs ──────────────────────────────────────────────────────

    def evaluate(self, action: dict) -> dict:
        """Call /v1/evaluate without running anything. Returns the verdict dict."""
        body: dict[str, Any] = {
            "agent": self.agent,
            "session_id": self.session_id,
            "session_goal": self.session_goal,
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
        self.last_verdict = verdict
        if not self.session_id and verdict.get("session_id"):
            self.session_id = verdict["session_id"]
        if verdict.get("request_id"):
            self.last_request_id = verdict["request_id"]
        return verdict

    def guard(self, action: dict, run: Callable[[dict], Any]) -> Any:
        """Wrap a tool execution with the policy gate.

        In `enforce` mode: returns the inner's result on allow, raises
        ClevrBlockedError on block, behaves per on_escalate on escalate.
        In `shadow` mode: ALWAYS runs the inner and returns a dict
        {result, verdict, shadow, would_have_blocked, would_have_escalated}.
        """
        verdict = self.evaluate(action)

        if self.mode == "shadow":
            return {
                "result": run(verdict),
                "verdict": verdict,
                "shadow": True,
                "would_have_blocked": verdict.get("effect") == "block",
                "would_have_escalated": verdict.get("effect") == "escalate",
            }

        effect = verdict.get("effect")
        if effect == "allow":
            return run(verdict)
        if effect == "block":
            raise ClevrBlockedError(verdict)
        # escalate
        if self.on_escalate == "allow":
            return run(verdict)
        if self.on_escalate == "wait":
            return self._wait_then_run(verdict, run)
        raise ClevrEscalatedError(verdict)

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

    def _wait_then_run(self, verdict: dict, run: Callable, timeout_s: float = 600.0) -> Any:
        """Poll the decision endpoint until the escalation is resolved by a human."""
        decision_id = verdict.get("decision_id")
        start = time.time()
        delay = 1.0
        while time.time() - start < timeout_s:
            time.sleep(delay)
            delay = min(delay * 1.5, 15.0)
            try:
                status = self._get(f"/v1/decisions/{decision_id}")
            except Exception:
                continue
            resolution = (status or {}).get("resolution")
            if resolution == "approved":
                return run(verdict)
            if resolution == "rejected":
                raise ClevrBlockedError({
                    **verdict,
                    "reason": f"Rejected by {status.get('resolved_by')}: {status.get('resolution_reason')}",
                })
        raise ClevrEscalatedError({**verdict, "reason": f"Escalation timed out after {int(timeout_s)}s"})


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
        try:
            from clevr_langchain import ClevrCallbackHandler  # type: ignore
            from langchain_core.callbacks import manager as _cbmgr  # type: ignore
            handler = ClevrCallbackHandler(clevr_instance)
            # LangChain has `register_configure_hook` for global handlers.
            # The shape changed across versions, so try a couple of paths.
            if hasattr(_cbmgr, "register_configure_hook"):
                _cbmgr.register_configure_hook("clevr_handler", lambda inheritable: handler)
            else:
                # Fallback: push into the default callback list if available.
                if hasattr(_cbmgr, "BaseCallbackManager"):
                    setattr(_cbmgr.BaseCallbackManager, "_clevr_handler", handler)
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
