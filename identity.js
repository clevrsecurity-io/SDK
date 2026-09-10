// @clevr/sdk — verifiable agent identity signer.
//
// An agent proves it is the workload it claims by signing each /v1/evaluate
// request with its own Ed25519 key. The engine verifies the signature and pins
// the public key on first signed sight (trust-on-first-use). This is the client
// half; it MUST produce a byte-identical signing string to the brain
// (src/lib/identity_proof.js) or the signature will not verify.
//
// Server-side only (uses node:crypto). Agents that don't provision a key keep
// working unsigned — their identity stays 'asserted'.
import crypto from 'node:crypto';

const PKCS8_ED25519_PREFIX = Buffer.from('302e020100300506032b657004220420', 'hex');
const SPKI_ED25519_PREFIX = Buffer.from('302a300506032b6570032100', 'hex');

// Deterministic, language-neutral. MUST match the brain byte-for-byte.
export function identitySigningString(f) {
  return [
    'clevr-agent-id-v1',
    f.agent || '',
    f.action_type || '',
    f.action || '',
    f.tool || '',
    f.session_id || '',
    String(f.ts || ''),
    f.nonce || '',
  ].join('\n');
}

/**
 * Sign an identity proof. `seedB64` is the agent's secret raw-32 Ed25519 seed
 * (base64); `bound` is { agent, action_type, action, tool, session_id }.
 * Returns { key, ts, nonce, sig } to attach as request body.identity.
 */
export function signIdentityProof(seedB64, bound, { nowSec = Math.floor(Date.now() / 1000), nonce } = {}) {
  const seed = Buffer.from(String(seedB64), 'base64');
  if (seed.length !== 32) throw new Error('signIdentityProof: seed must be 32 raw bytes (base64)');
  const priv = crypto.createPrivateKey({ key: Buffer.concat([PKCS8_ED25519_PREFIX, seed]), format: 'der', type: 'pkcs8' });
  const spki = crypto.createPublicKey(priv).export({ format: 'der', type: 'spki' });
  const key = spki.subarray(spki.length - 32).toString('base64');
  const n = nonce || crypto.randomBytes(9).toString('base64url');
  const sig = crypto.sign(null, Buffer.from(identitySigningString({ ...bound, ts: nowSec, nonce: n }), 'utf8'), priv);
  return { key, ts: nowSec, nonce: n, sig: 'ed25519:' + sig.toString('base64') };
}

/**
 * Generate a fresh identity keypair for provisioning a workload. Keep `seed`
 * secret (pass it as CLEVR_IDENTITY_SEED or opts.identitySeed); `key` is what
 * Clevr pins on first sight.
 */
export function generateIdentitySeed() {
  const { privateKey } = crypto.generateKeyPairSync('ed25519');
  const pkcs8 = privateKey.export({ format: 'der', type: 'pkcs8' });
  const spki = crypto.createPublicKey(privateKey).export({ format: 'der', type: 'spki' });
  return {
    seed: pkcs8.subarray(pkcs8.length - 32).toString('base64'),
    key: spki.subarray(spki.length - 32).toString('base64'),
  };
}
