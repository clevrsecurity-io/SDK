// verify-credential.mjs: check a credential Clevr minted, as the service that
// receives it would (docs/api/credentials.mdx). No Clevr secret, no API key:
// the workspace's public key from /v1/issuers/:org/jwks.json, then
// /v1/introspect to learn whether it still stands.
//
//   CLEVR_URL=https://clevr.example.com node sdk/examples/verify-credential.mjs <token> [resource] [access]
//
// Prints what was checked and exits 0 when the credential may be honoured.
import crypto from 'node:crypto'

let CLEVR = process.env.CLEVR_URL || 'http://localhost:8081'
while (CLEVR.endsWith('/')) CLEVR = CLEVR.slice(0, -1)
const [token, resource, access] = process.argv.slice(2)
if (!token) { console.error('usage: verify-credential.mjs <token> [resource] [access]'); process.exit(2) }

const RANK = { limited: 0, read: 1, write: 2, admin: 3 }
const no = (why) => { console.log(`refuse: ${why}`); process.exit(1) }

const [h, p, s] = token.split('.')
let header, claims
try { header = JSON.parse(Buffer.from(h, 'base64url')); claims = JSON.parse(Buffer.from(p, 'base64url')) } catch { no('unreadable token') }
if (header.alg !== 'EdDSA' || header.typ !== 'clevr-credential+jwt') no(`not a Clevr credential (alg ${header.alg}, typ ${header.typ})`)

const org = String(claims.iss || '').replace(/^clevr:/, '')
const jwks = await fetch(`${CLEVR}/v1/issuers/${encodeURIComponent(org)}/jwks.json`).then((r) => (r.ok ? r.json() : null)).catch(() => null)
const jwk = jwks?.keys?.find((k) => k.kid === header.kid)
if (!jwk) no(`no key for ${claims.iss}`)
const key = crypto.createPublicKey({ key: jwk, format: 'jwk' })
if (!crypto.verify(null, Buffer.from(`${h}.${p}`), key, Buffer.from(s, 'base64url'))) no('signature does not verify')
console.log(`signature: ok, signed by ${claims.iss} (kid ${header.kid})`)

if (!Number.isFinite(claims.exp) || claims.exp <= Date.now() / 1000) no('expired')
console.log(`expiry: ok, ${Math.round(claims.exp - Date.now() / 1000)} s left`)

if (resource || access) {
  const [target, granted] = String(claims.scope || '').split(':')
  if (target !== '*' && target !== resource) no(`scope is for ${target}, not ${resource}`)
  if ((RANK[granted] ?? -1) < (RANK[access || 'read'] ?? 99)) no(`scope grants ${granted}, ${access || 'read'} needed`)
  console.log(`scope: ok, ${claims.scope}`)
}

const st = await fetch(`${CLEVR}/v1/introspect`, {
  method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ token }),
}).then((r) => r.json()).catch(() => null)
if (!st) no('could not ask whether it still stands')
if (!st.active) no(`no longer stands: ${st.reason}${st.stopped_by ? ` (stopped by ${st.stopped_by})` : ''}`)
console.log(`still stands: ok, checked ${st.checked.join(', ')}`)
console.log(`agent ${claims.sub}, session ${claims.sid || 'none'}${claims.cnf ? ' (proven)' : ''}, decision ${claims.decision}`)
console.log('honour: yes')
