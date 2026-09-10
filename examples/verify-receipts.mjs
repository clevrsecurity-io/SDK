// verify-receipts.mjs — INDEPENDENT, OFFLINE verifier for a Clevr audit package.
//
//   node verify-receipts.mjs <evidence-package.json>
//
// An auditor runs this with ZERO trust in Clevr: it re-derives every receipt
// hash and verifies every Ed25519 signature. If a single byte of any decision
// was altered after signing — or a decision was inserted/reordered — the chain
// breaks and this reports exactly where. No network, no Clevr code, only Node's
// stdlib crypto.
//
// TRUST ANCHOR: the key inside the package only proves the receipts are
// self-consistent. To prove they are a specific ORG's receipts — and to detect a
// wholesale re-sign with a fresh key, a dropped (redacted) row, or a truncated
// tail — pass the org's independently-published public key:
//   node verify-receipts.mjs <package.json> --expected-key=<base64>
// Without it, the report is "internally consistent" but never "intact".
//
// It mirrors brain/src/lib/crypto.js exactly:
//   content_hash  = sha256( canonicalJSON(receipt_payload, seq:0) )
//   receipt_hash  = sha256( prev_hash + content_hash )
//   signature     = Ed25519( receipt_hash )  verified with the org public key
//   chain         : decision[i].prev_hash === decision[i-1].receipt_hash

import crypto from 'node:crypto'
import { readFileSync } from 'node:fs'
import { pathToFileURL } from 'node:url'

const GENESIS = '0'.repeat(64)
const sha256 = (s) => crypto.createHash('sha256').update(s, 'utf8').digest('hex')

// Build an Ed25519 public KeyObject from a raw 32-byte base64 key (RFC 8410 SPKI).
export function publicKeyFromB64 (b64) {
  const raw = Buffer.from(b64, 'base64')
  if (raw.length !== 32) throw new Error('public key must be 32 raw bytes (base64)')
  const spki = Buffer.concat([Buffer.from('302a300506032b6570032100', 'hex'), raw])
  return crypto.createPublicKey({ key: spki, format: 'der', type: 'spki' })
}

// Canonical content hash — identical field set + order to lib/crypto.js::contentHash,
// computed over the receipt_payload with seq forced to 0. JSON.stringify omits
// undefined-valued keys, exactly as the signer did.
export function contentHash (payload) {
  const ts = new Date(payload.created_at).toISOString()
  const canonical = JSON.stringify({
    org_id: payload.org_id,
    agent_id: payload.agent_id,
    // The conditional keys below MUST match lib/crypto.js::contentHash EXACTLY —
    // same keys, same order — or a receipt carrying any of them would false-flag
    // as tampered. Each is included only when present (JSON.stringify drops
    // undefined), exactly as the signer does.
    ...(payload.on_behalf_of ? { on_behalf_of: payload.on_behalf_of } : {}),
    ...(payload.identity_verified ? { identity_verified: true } : {}),
    ...(payload.input_verified ? { input_verified: true } : {}),
    seq: 0,
    action_type: payload.action_type,
    action: payload.action,
    effect: payload.effect,
    risk_level: payload.risk_level,
    reason: payload.reason,
    matched_policy: payload.matched_policy,
    ...(payload.tool ? { tool: payload.tool } : {}),
    ...(payload.target ? { target: payload.target } : {}),
    ...(payload.environment ? { environment: payload.environment } : {}),
    created_at: ts
  })
  return sha256(canonical)
}

export function verifyBundle (pkg, { expectedKeyB64 = null, expectedSeqLast = null } = {}) {
  const pubB64 = pkg?.public_key?.value
  if (!pubB64) throw new Error('package has no public_key.value')
  const pub = publicKeyFromB64(pubB64)
  // Trust anchor. The key embedded in the package proves the receipts are
  // INTERNALLY consistent — nothing more. To prove they are a specific ORG's,
  // the auditor must pin that org's real public key out-of-band (its published
  // key, or a key from a prior trusted export) and pass it in. Without a pin, or
  // on a mismatch, we refuse the strong "intact" claim: otherwise a producer
  // could sign a fully fabricated history with a freshly generated key and it
  // would verify against itself.
  const keyFingerprint = sha256(Buffer.from(pubB64, 'base64')).slice(0, 32)
  const keyPinned = expectedKeyB64 != null
  let keyMatches = false
  if (keyPinned) {
    try { keyMatches = Buffer.from(pubB64, 'base64').equals(Buffer.from(expectedKeyB64, 'base64')) } catch { keyMatches = false }
  }
  const decisions = pkg.decisions || []

  let prev = decisions.length ? decisions[0].prev_hash : GENESIS
  let brokenAt = null, missingSeq = null, redacted = 0
  let expectedSeq = null
  let ok = 0
  const failures = []

  for (const d of decisions) {
    const seqNum = Number(d.seq)
    // A seq gap = a WINDOWED export (from/to) or a lawfully-removed row, NOT
    // tampering. The surviving signatures still verify; we only flag a real
    // signature failure, or a linkage break with NO accompanying gap.
    const seqGap = expectedSeq !== null && seqNum !== expectedSeq
    if (seqGap && missingSeq === null) missingSeq = expectedSeq
    // Chain link: this decision's prev_hash must equal the previous decision's
    // sealed receipt_hash — so a reorder/insertion breaks the chain.
    const linkOk = d.prev_hash === prev

    if (d.redacted) {
      // A redacted row carries no signature and no content, so it CANNOT be
      // cryptographically verified — only its chain link is checkable. A bare
      // `redacted` flag is not a signed tombstone: a producer can blank any row
      // (dropping an incriminating decision) and the link still matches. So we
      // check the link, but we do NOT count it as verified, and its presence
      // downgrades the strong "intact" claim below.
      redacted++
      if (!linkOk && !seqGap) { if (brokenAt === null) brokenAt = seqNum; failures.push({ seq: seqNum, decision_id: d.decision_id, redacted: true, chain_link_intact: false }) }
    } else {
      // The cryptographic anchor: receipt_hash = sha256(prev_hash + contentHash(payload)),
      // signed with Ed25519. Any post-signing byte change fails here.
      const rH = sha256((d.prev_hash ?? '') + contentHash(d.receipt_payload))
      let sigOk = false
      try {
        const sig = Buffer.from(String(d.signature).replace(/^ed25519:/, ''), 'base64url')
        sigOk = crypto.verify(null, Buffer.from(rH, 'utf8'), pub, sig)
      } catch { sigOk = false }
      if (!sigOk) { if (brokenAt === null) brokenAt = seqNum; failures.push({ seq: seqNum, decision_id: d.decision_id, signature_valid: false, chain_link_intact: linkOk }) }
      else if (!linkOk && !seqGap) { if (brokenAt === null) brokenAt = seqNum; failures.push({ seq: seqNum, decision_id: d.decision_id, signature_valid: true, chain_link_intact: false }) }
      else ok++
    }
    prev = d.receipt_hash
    expectedSeq = seqNum + 1
  }

  // HEAD-completeness: the first row must be genesis-anchored (prev_hash ===
  // GENESIS). If it isn't, rows were dropped from the FRONT — every surviving
  // signature still verifies, so only this check catches a head truncation.
  const headComplete = decisions.length > 0 && decisions[0].prev_hash === GENESIS
  // TAIL-completeness cannot be proven from the package alone: dropping the LAST
  // k rows leaves every remaining signature valid and every link intact. It needs
  // an out-of-band anchor — the org's independently-known latest seq (published
  // the same way as its public key). tailProven is true only when that anchor is
  // supplied AND matches the last row's seq.
  const lastSeq = decisions.length ? Number(decisions[decisions.length - 1].seq) : null
  const tailAnchored = expectedSeqLast != null
  const tailProven = tailAnchored && lastSeq != null && lastSeq === Number(expectedSeqLast)

  // `signaturesValid`: no signature or link break among the checkable rows.
  // `intact` (the STRONG claim) additionally requires: the key pinned + matching,
  // no redacted (cryptographically-unverifiable) rows, no sequence gaps, the head
  // genesis-anchored (no front truncation), and the tail anchored + matching (no
  // back truncation). Anything short is reported as internally-consistent-but-
  // qualified, never as fully intact.
  const signaturesValid = brokenAt === null
  const intact = signaturesValid && keyPinned && keyMatches && redacted === 0 &&
    missingSeq === null && headComplete && tailProven
  return {
    total: decisions.length, verified: ok, signaturesValid, intact,
    broken_at: brokenAt, missing_seq: missingSeq, redacted, failures,
    key_pinned: keyPinned, key_matches: keyMatches, key_fingerprint: keyFingerprint,
    head_complete: headComplete, tail_anchored: tailAnchored, tail_proven: tailProven,
    last_seq: lastSeq, expected_seq_last: expectedSeqLast != null ? Number(expectedSeqLast) : null
  }
}

// Print a human-readable report for a verified package. Returns the process
// exit code (0 = intact, 1 = tampered) so a caller can decide when to exit.
// Reused by the `clevr verify <bundle.json>` CLI so the offline output is
// identical whether run as this single file or through the packaged SDK.
export function printReport (pkg, r = verifyBundle(pkg)) {
  console.log('Clevr audit package: independent verification')
  console.log('  org       :', pkg.meta?.org_name, `(${pkg.meta?.org_id})`)
  console.log('  regulation:', pkg.meta?.regulation || '(all decisions)')
  console.log('  period    :', JSON.stringify(pkg.meta?.period))
  console.log('  decisions :', r.total, `(seq ${pkg.meta?.seq_first}…${pkg.meta?.seq_last})`)
  console.log('  verified  :', r.verified, '/', r.total)
  console.log('  key       :', r.key_fingerprint,
    r.key_pinned ? (r.key_matches ? '(matches pinned key)' : '(DOES NOT MATCH pinned key)') : '(NOT pinned — pass --expected-key to prove ownership)')
  if (r.redacted) console.log('  redacted  :', r.redacted, '(content not cryptographically verifiable)')
  if (r.missing_seq != null) console.log('  gap       : sequence gap from seq', r.missing_seq, '(could be a windowed export OR a removed/truncated row)')
  console.log('  head      :', r.head_complete ? 'genesis-anchored (no front truncation)' : 'NOT genesis-anchored (rows dropped from the front, or a windowed export)')
  console.log('  tail      :', r.tail_proven ? `anchored at seq ${r.expected_seq_last} (no back truncation)`
    : r.tail_anchored ? `MISMATCH: last row is seq ${r.last_seq}, expected ${r.expected_seq_last}`
      : `ends at seq ${r.last_seq} (pass --expected-seq-last to prove no rows were truncated after it)`)

  // An actual signature or link break is tampering — always fail.
  if (!r.signaturesValid) {
    console.log('\n  ❌ TAMPERED: chain breaks at seq', r.broken_at)
    console.log(JSON.stringify(r.failures.slice(0, 5), null, 2))
    return 1
  }
  // Signatures verify, key is pinned + matches, nothing redacted, no gaps.
  if (r.intact) {
    console.log('\n  ✅ AUDIT CHAIN INTACT. Every signature valid, every link unbroken, key matches the pinned org key.')
    console.log('     Tamper-evident: any post-signing alteration would have failed here.')
    return 0
  }
  // Signatures verify, but the STRONG claim cannot be made. Say exactly why —
  // never print "INTACT" here.
  console.log('\n  ⚠️  INTERNALLY CONSISTENT, BUT NOT INDEPENDENTLY PROVEN:')
  if (!r.key_pinned) console.log('     - the signing key is taken from the package itself; pin the org\'s published key with --expected-key to prove these are its receipts.')
  else if (!r.key_matches) console.log('     - the package key does NOT match the pinned org key: these receipts were signed by a different key.')
  if (r.redacted) console.log('     - ' + r.redacted + ' redacted row(s): their content and signature are gone, so a dropped decision cannot be distinguished from a lawful erasure.')
  if (r.missing_seq != null) console.log('     - a sequence gap is present: a tail truncation or a removed row cannot be ruled out without the org\'s chain-head anchor.')
  if (!r.head_complete) console.log('     - the first row is not genesis-anchored: rows may have been dropped from the FRONT of the chain (or this is a deliberately windowed export).')
  if (r.head_complete && !r.tail_anchored) console.log('     - tail-completeness is unproven: dropping the LAST rows leaves every surviving signature valid. Pass --expected-seq-last=<n> (the org\'s independently-known latest seq) to rule out a back truncation.')
  if (r.tail_anchored && !r.tail_proven) console.log('     - the last row seq does not match the expected latest seq: rows were truncated from the END of the chain.')
  return 3
}

// ── main ──────────────────────────────────────────────────────────────────
// Runs ONLY when this file is executed directly (node verify-receipts.mjs …),
// not when it is imported (e.g. by the `clevr` CLI). Keeps the file usable as a
// single, dependency-free artifact an auditor can copy and run in isolation.
if (import.meta.url === pathToFileURL(process.argv[1] || '').href) {
  const args = process.argv.slice(2)
  // Optional out-of-band trust anchor: --expected-key=<base64> (or env
  // CLEVR_EXPECTED_PUBKEY) is the org's independently-published public key. Pass
  // it to get the strong "intact" claim; without it the report stays qualified.
  let expectedKeyB64 = process.env.CLEVR_EXPECTED_PUBKEY || null
  // Out-of-band tail anchor: the org's independently-known latest seq (published
  // alongside its public key). Proves no rows were truncated from the END.
  let expectedSeqLast = process.env.CLEVR_EXPECTED_SEQ_LAST || null
  const positional = []
  for (const a of args) {
    let m
    if ((m = a.match(/^--expected-key=(.+)$/))) expectedKeyB64 = m[1]
    else if ((m = a.match(/^--expected-seq-last=(\d+)$/))) expectedSeqLast = m[1]
    else positional.push(a)
  }
  const path = positional[0]
  if (!path) { console.error('usage: node verify-receipts.mjs <evidence-package.json> [--expected-key=<base64>] [--expected-seq-last=<n>]'); process.exit(2) }
  const pkg = JSON.parse(readFileSync(path, 'utf8'))
  process.exit(printReport(pkg, verifyBundle(pkg, { expectedKeyB64, expectedSeqLast })))
}
