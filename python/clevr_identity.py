"""clevr identity — verifiable agent identity signer (Python).

Mirror of the Node ``@clevr/sdk`` ``identity.js`` and the brain's
``src/lib/identity_proof.js``. An agent proves it is the workload it claims by
signing each ``/v1/evaluate`` request with its own Ed25519 key; the engine
verifies the signature and pins the public key on first signed sight
(trust-on-first-use), so identity goes from *asserted* to *verified*.

The signing string is language-neutral and MUST match the brain byte-for-byte,
or the signature will not verify. Signing needs the ``cryptography`` package for
Ed25519 — an **optional** dependency: the SDK core stays pure-stdlib and keeps
working unsigned without it (identity stays 'asserted'). Only this opt-in
signed-identity path requires it.
"""

from __future__ import annotations

import base64
import secrets
import time
from typing import Optional


def identity_signing_string(f: dict) -> str:
    """Deterministic, language-neutral. MUST match the brain + Node byte-for-byte."""
    return "\n".join([
        "clevr-agent-id-v1",
        str(f.get("agent") or ""),
        str(f.get("action_type") or ""),
        str(f.get("action") or ""),
        str(f.get("tool") or ""),
        str(f.get("session_id") or ""),
        str(f.get("ts") or ""),
        str(f.get("nonce") or ""),
    ])


def _ed25519():
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import ed25519
        return ed25519, serialization
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ImportError(
            "clevr: signed agent identity needs the 'cryptography' package "
            "(pip install cryptography). Without it the agent runs unsigned and "
            "its identity stays 'asserted'."
        ) from e


def sign_identity_proof(
    seed_b64: str,
    bound: dict,
    now_sec: Optional[int] = None,
    nonce: Optional[str] = None,
) -> dict:
    """Sign an identity proof.

    Args:
        seed_b64: the agent's secret raw-32 Ed25519 seed (base64).
        bound:    {agent, action_type, action, tool, session_id} — the fields the
                  proof is bound to, so a captured proof can't be lifted onto a
                  different agent or tool.
    Returns ``{key, ts, nonce, sig}`` to attach as request ``body['identity']``.
    """
    ed25519, serialization = _ed25519()
    seed = base64.b64decode(seed_b64)
    if len(seed) != 32:
        raise ValueError("sign_identity_proof: seed must be 32 raw bytes (base64)")
    priv = ed25519.Ed25519PrivateKey.from_private_bytes(seed)
    raw_pub = priv.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    key = base64.b64encode(raw_pub).decode("ascii")
    ts = int(now_sec if now_sec is not None else time.time())
    n = nonce or base64.urlsafe_b64encode(secrets.token_bytes(9)).decode("ascii").rstrip("=")
    msg = identity_signing_string({**bound, "ts": ts, "nonce": n}).encode("utf-8")
    sig = priv.sign(msg)
    return {
        "key": key,
        "ts": ts,
        "nonce": n,
        "sig": "ed25519:" + base64.b64encode(sig).decode("ascii"),
    }


def generate_identity_seed() -> dict:
    """Generate a fresh identity keypair for provisioning a workload.

    Keep ``seed`` secret (pass it as ``CLEVR_IDENTITY_SEED`` or
    ``identity_seed=``); ``key`` is what Clevr pins on first sight.
    """
    ed25519, serialization = _ed25519()
    priv = ed25519.Ed25519PrivateKey.generate()
    seed = priv.private_bytes(
        serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
        serialization.NoEncryption())
    raw_pub = priv.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return {
        "seed": base64.b64encode(seed).decode("ascii"),
        "key": base64.b64encode(raw_pub).decode("ascii"),
    }
