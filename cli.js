#!/usr/bin/env node
// @clevr/sdk CLI — `npx @clevr/sdk <command>`
//
// Commands:
//   policy push <file.yaml|file.json>   Upload a Clevr policy doc
//   policy lint <file.yaml|file.json>   Parse + validate without uploading
//   verify <decision-id>                Verify a decision via the engine (needs a key)
//   verify <evidence-package.json>      Verify an exported audit bundle OFFLINE (no key)
//   help                                Show this help
//
// ENV: CLEVR_URL (default http://localhost:8787)
//      CLEVR_API_KEY (required for push)
//
// YAML parsing uses js-yaml — battle-tested CommonMark-strict implementation
// that handles anchors, multi-doc streams, block literals, and the full
// edge-case surface customers can throw at us. The hand-rolled subset
// parser we shipped in v0.2.0 was fine for the docs example but failed on
// anything with anchors or multi-line block scalars — replaced here.

import fs from 'node:fs'
import path from 'node:path'
import { printReport } from './examples/verify-receipts.mjs'
// js-yaml is imported lazily inside parsePolicy() so that `verify` and `help`
// (the offline, zero-trust paths) run with only Node's standard library.

const BASE = (process.env.CLEVR_URL || 'http://localhost:8787').replace(/\/$/, '')
const KEY  = process.env.CLEVR_API_KEY

const [, , cmd, ...rest] = process.argv

if (cmd === 'help' || !cmd) {
  printHelp(); process.exit(0)
}
if (cmd === 'policy') {
  const sub = rest[0]
  if (sub === 'push' || sub === 'lint') {
    const file = rest[1]
    if (!file) die(`Missing file path. Usage: clevr policy ${sub} <file>`)
    const raw = fs.readFileSync(path.resolve(file), 'utf-8')
    const doc = file.endsWith('.json') ? JSON.parse(raw) : await parsePolicy(raw)
    normalizeShorthandRules(doc)
    validate(doc)
    if (sub === 'lint') {
      console.log('✓ Valid policy document.')
      console.log(`  Agent:        ${doc.agent}`)
      console.log(`  Default:      ${doc.default || 'deny'}`)
      console.log(`  Rules:        ${(doc.rules || []).length}`)
      console.log(`  Rate limits:  ${Object.keys(doc.rate_limits || {}).length}`)
      console.log(`  Budgets:      ${Object.keys(doc.budget || {}).length}`)
      process.exit(0)
    }
    if (!KEY) die('CLEVR_API_KEY is required to push (set in env).')
    await push(doc)
    process.exit(0)
  }
  die(`Unknown policy sub-command: ${sub}`)
}
if (cmd === 'verify') {
  const arg = rest[0]
  if (!arg) die('Usage: clevr verify <decision-id | evidence-package.json>')
  // A path to an exported evidence bundle → verify OFFLINE (no key, no engine,
  // zero trust): re-derive every receipt hash and check every Ed25519 signature
  // using only the public key inside the bundle. Anything else is treated as a
  // decision id and verified through the engine's /verify route.
  const asFile = path.resolve(arg)
  if (/\.json$/i.test(arg) || fs.existsSync(asFile)) {
    let pkg
    try { pkg = JSON.parse(fs.readFileSync(asFile, 'utf-8')) }
    catch (e) { die(`Cannot read evidence package "${arg}": ${e.message}`) }
    process.exit(printReport(pkg))
  }
  await verify(arg)
  process.exit(0)
}
die(`Unknown command: ${cmd}. Run \`clevr help\` for usage.`)

/* ───────────────────────────────── helpers ───────────────────────────── */

async function push (doc) {
  const r = await fetch(`${BASE}/brain/api/policies/import`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json',
               Authorization: 'Bearer ' + KEY },
    body: JSON.stringify(doc)
  })
  const body = await r.json().catch(() => ({}))
  if (!r.ok) die(`Push failed (${r.status}): ${JSON.stringify(body)}`)
  console.log(`✓ Pushed policy ${body.policy?.name} (${body.policy?.id})`)
  console.log(`  Rules:       ${body.summary?.rules}`)
  console.log(`  Rate limits: ${body.summary?.rate_limits}`)
  console.log(`  Budgets:     ${body.summary?.budgets}`)
  console.log(`  Default:     ${body.summary?.default}`)
}

async function verify (decId) {
  const r = await fetch(`${BASE}/v1/decisions/${decId}/verify`, {
    method: 'GET',
    headers: { Authorization: 'Bearer ' + KEY }
  })
  const body = await r.json().catch(() => ({}))
  if (!r.ok) die(`Verify failed (${r.status}): ${JSON.stringify(body)}`)
  console.log(`Decision ${body.id} (seq ${body.seq})`)
  console.log(`  Signature valid: ${body.signature_valid ? '✓' : '✗'}`)
  console.log(`  Linkage  valid: ${body.linkage_valid   ? '✓' : '✗'}`)
  console.log(`  Tamper-evident: ${body.tamper_evident  ? '✓' : '✗'}`)
}

function validate (doc) {
  if (!doc || typeof doc !== 'object') die('Document must be a mapping at root.')
  if (!doc.agent || typeof doc.agent !== 'string')
    die('Field `agent` (string) is required.')
  if (doc.version && doc.version !== 1)
    die(`Unsupported policy version: ${doc.version} (only v1).`)
  const VERDICTS = new Set(['permit', 'defer', 'deny', 'allow', 'escalate', 'block'])
  if (doc.default && !VERDICTS.has(doc.default))
    die(`Invalid default verdict: ${doc.default}.`)
  for (const r of (doc.rules || [])) {
    if (!r.verdict || !VERDICTS.has(r.verdict))
      die(`Rule must have verdict in ${[...VERDICTS].join('|')}: ${JSON.stringify(r)}`)
    if (!r.tool && !r.action_type)
      die(`Rule must have at least one of {tool, action_type}: ${JSON.stringify(r)}`)
  }
}

function printHelp () {
  console.log(`@clevr/sdk CLI

Usage:
  clevr policy push <file.yaml|.json>   Upload a Clevr policy document
  clevr policy lint <file.yaml|.json>   Parse + validate without uploading
  clevr verify     <decision-id>        Verify one decision via the engine (needs a key)
  clevr verify     <evidence.json>      Verify an exported audit bundle OFFLINE (no key, no engine)
  clevr help                             Show this help

Environment:
  CLEVR_URL       Engine base URL  (default http://localhost:8787)
  CLEVR_API_KEY   Bearer key (required for push and for online verify;
                  NOT needed for offline bundle verification)

Example clevr.yaml:
  version: 1
  agent:   support-bot
  default: deny
  rules:
    - permit: kb.search
    - permit: ticket.update
    - defer:  email.send
      if:    target.domain != "@acme.com"
    - deny:  delete.*
  rate_limits:
    email.send: { max: 50, per: hour }
  budget:
    daily: { max_eur: 20, on_exceed: defer }
`)
}

function die (msg) {
  process.stderr.write(`clevr: ${msg}\n`)
  process.exit(1)
}

/* ─────────────────────────── Policy/YAML parser ───────────────────────────
   Thin wrapper around js-yaml's safe loader. We accept the full YAML 1.2
   surface and then normalize the policy shorthand (`- permit: x`)
   into the canonical {verdict, tool} shape downstream code expects. */
async function parsePolicy (text) {
  let yamlLoad
  try { ({ load: yamlLoad } = await import('js-yaml')) }
  catch { die('YAML support needs the js-yaml package. Install it, or convert the policy to .json.') }
  let doc
  try {
    doc = yamlLoad(text, { schema: undefined, json: false })
  } catch (e) {
    die(`YAML parse error: ${e.message}`)
  }
  if (doc == null || typeof doc !== 'object') {
    die('Document must be a YAML mapping at the root.')
  }
  return doc
}

function normalizeShorthandRules (doc) {
  if (Array.isArray(doc.rules)) {
    doc.rules = doc.rules.map((r) => normalizeRule(r))
  }
}

function normalizeRule (r) {
  // Shorthand: `{ permit: 'kb.search' }` → `{ verdict: 'permit', tool: 'kb.search' }`
  if (!r || typeof r !== 'object') return r
  for (const v of ['permit', 'defer', 'deny', 'allow', 'escalate', 'block']) {
    if (v in r && r[v] != null) {
      const out = { verdict: v, tool: r[v], ...r }
      delete out[v]
      return out
    }
  }
  return r
}
