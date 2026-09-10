"""Minimal demo of the Python Clevr SDK (no LangChain dependency).

Run:
    export CLEVR_API_KEY=clevr_sk_...
    python sdk/python/examples/basic_demo.py
"""
from __future__ import annotations

import os
import sys

# Allow importing clevr.py from the parent dir when running standalone.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from clevr import Clevr, chain, ClevrBlockedError


def main() -> None:
    clevr = Clevr(agent="demo-researcher", mode="shadow")
    clevr.with_root_human("alice@acme.com", display="Alice · Demo")

    # Action 1 — safe read, would be allowed
    r1 = clevr.guard(
        {"tool": "kb.search", "action_type": "read", "target": "kb/incidents",
         "action": "search incident archive"},
        run=lambda v: f"[would_run] kb.search effect={v['effect']}",
    )
    print("kb.search shadow →", r1["verdict"]["effect"], "·", r1["result"])

    # Action 2 — destructive on an ungoverned agent — would be blocked by verb-aware
    r2 = clevr.guard(
        {"tool": "db.purge", "action_type": "delete", "target": "customers",
         "action": "attempt to purge customer database"},
        run=lambda v: f"[would_run] db.purge effect={v['effect']}",
    )
    print("db.purge shadow  →", r2["verdict"]["effect"], "would_have_blocked=", r2["would_have_blocked"])

    # Spawn a child sub-agent and guard via it — the chain extends automatically
    child_clevr = clevr.child("delegated-investigator")
    print("child agent chain:", [h["id"] for h in child_clevr.actor_chain])
    r3 = child_clevr.guard(
        {"tool": "kb.read", "action_type": "read", "target": "kb/runbooks",
         "action": "read runbook"},
        run=lambda v: f"[would_run] kb.read by sub-agent effect={v['effect']}",
    )
    print("kb.read (child) →", r3["verdict"]["effect"], "chain_depth=", r3["verdict"]["chain_depth"])


if __name__ == "__main__":
    if not os.environ.get("CLEVR_API_KEY"):
        print("Set CLEVR_API_KEY first.", file=sys.stderr)
        sys.exit(1)
    try:
        main()
    except ClevrBlockedError as e:
        print(f"Blocked: {e}", file=sys.stderr)
        sys.exit(1)
