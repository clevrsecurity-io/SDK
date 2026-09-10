"""Tests for clevr verifiable agent identity (Python).

Run: python3 test_identity.py   (from sdk/python/)

The signing-string test is pure stdlib and always runs — it pins the wire format
byte-for-byte, which is what makes the Python signature verify against the brain
(src/lib/identity_proof.js) and the Node SDK (identity.js). The sign/verify tests
need the optional 'cryptography' package and skip cleanly without it.
"""

import base64
import sys

sys.path.insert(0, ".")
from clevr_identity import identity_signing_string  # noqa: E402


def test_signing_string_is_byte_exact():
    # Language-neutral scalar join — MUST equal the brain + Node output exactly.
    s = identity_signing_string({
        "agent": "a", "action_type": "tool_call", "action": "x",
        "tool": "t", "session_id": "s", "ts": 123, "nonce": "n",
    })
    assert s == "clevr-agent-id-v1\na\ntool_call\nx\nt\ns\n123\nn", repr(s)


def test_signing_string_handles_missing_fields():
    # Absent fields render as empty segments (never "None"), so an unsigned-shaped
    # bound still lines up with the brain's own empty-field handling.
    s = identity_signing_string({"agent": "a", "ts": 1})
    assert s == "clevr-agent-id-v1\na\n\n\n\n\n1\n", repr(s)


def _has_crypto():
    try:
        import cryptography  # noqa: F401
        return True
    except ImportError:
        return False


def test_sign_verify_and_binding():
    if not _has_crypto():
        print("  (skipped sign tests — 'cryptography' not installed)")
        return
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    from clevr_identity import generate_identity_seed, sign_identity_proof

    kp = generate_identity_seed()
    bound = {"agent": "py", "action_type": "tool_call", "action": "kb.search",
             "tool": "kb.search", "session_id": "s1"}
    proof = sign_identity_proof(kp["seed"], bound)

    assert proof["key"] == kp["key"], "presented key must equal the seed-derived pubkey"
    assert proof["sig"].startswith("ed25519:")

    pub = Ed25519PublicKey.from_public_bytes(base64.b64decode(proof["key"]))
    sig = base64.b64decode(proof["sig"][len("ed25519:"):])
    good = identity_signing_string({**bound, "ts": proof["ts"], "nonce": proof["nonce"]}).encode()
    pub.verify(sig, good)  # raises InvalidSignature if wrong

    # Binding: the same proof must NOT verify for a different agent — a captured
    # proof can't be lifted onto another workload.
    lifted = identity_signing_string({**bound, "agent": "other", "ts": proof["ts"], "nonce": proof["nonce"]}).encode()
    try:
        pub.verify(sig, lifted)
    except InvalidSignature:
        pass
    else:
        raise AssertionError("proof for 'py' wrongly verified as 'other-agent'")


if __name__ == "__main__":
    n = 0
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            n += 1
            print(f"  ✓ {_name}")
    print(f"\nclevr identity (python): {n} passed")
