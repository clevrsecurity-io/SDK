// Offline check of what the gates reported for one decision
// (GET /v1/decisions/:id/enforcement, every page joined: ?after=<next_after>
// until next_after is null), with nothing but the workspace's public key:
// order, links, hashes and signatures. Kept apart from the brain's code on
// purpose; the canonical string is the brain's (lib/ledger/enforcement_reports.js).
// A valid history proves what Clevr was told and when, not that the target did it.
import crypto from 'node:crypto'
import { publicKeyFromB64 } from './examples/verify-receipts.mjs'

const GENESIS = '0'.repeat(64)

export function enforcementCanonical (e) {
  return JSON.stringify(['clevr-enforcement-v1', e.id, e.org_id, e.decision_id, Number(e.seq), e.kind,
    e.enforced, e.received_at ? new Date(e.received_at).toISOString() : null,
    e.reporter ?? null, e.report_id ?? null, e.prev_hash])
}

// expectedKeyB64 and expectedDecisionId come from you, not from the response:
// a history checked against the key it brought with it proves nothing. Pass
// expectedTailHash (the last event_hash you saw earlier) to know nothing was
// dropped from the end since.
export function verifyEnforcementHistory (events, { expectedKeyB64, expectedDecisionId, expectedTailHash = null } = {}) {
  if (!expectedKeyB64 || !expectedDecisionId) return { valid: false, error: 'expectedKeyB64 and expectedDecisionId are required' }
  if (!Array.isArray(events) || !events.length) return { valid: false, error: 'no reports' }
  let key
  try { key = publicKeyFromB64(expectedKeyB64) } catch { return { valid: false, error: 'bad public key' } }
  let prev = GENESIS
  for (let i = 0; i < events.length; i++) {
    const e = events[i]
    const at = { valid: false, broken_at: i + 1 }
    if (!e || typeof e !== 'object') return { ...at, error: 'malformed report' }
    if (e.decision_id !== expectedDecisionId) return { ...at, error: 'another decision' }
    if (Number(e.seq) !== i + 1 || e.prev_hash !== prev) return { ...at, error: 'order or link' }
    let hash
    try { hash = crypto.createHash('sha256').update(enforcementCanonical(e), 'utf8').digest('hex') } catch { return { ...at, error: 'malformed report' } }
    const sig = String(e.signature || '')
    const ok = hash === e.event_hash && sig.startsWith('ed25519:') &&
      crypto.verify(null, Buffer.from(hash, 'utf8'), key, Buffer.from(sig.slice(8), 'base64url'))
    if (!ok) return { ...at, error: 'hash or signature' }
    prev = hash
  }
  return { valid: true, checked: events.length, tail_hash: prev,
    complete: expectedTailHash ? events.some((e) => e.event_hash === expectedTailHash) : null }
}
