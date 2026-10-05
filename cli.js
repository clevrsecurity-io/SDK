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
      console.log(`  Default:      ${doc.default || 'none'}`)
      console.log(`  Rules:        ${(doc.rules || []).length}`)
      // lint never reaches the engine, so it says itself what a push is told.
      for (const n of [...defaultNotes(doc), ...limitNotes(doc)]) console.log(`  Note: ${n}`)
      process.exit(0)
    }
    if (!KEY) die('CLEVR_API_KEY is required to push: set it to a console session token.')
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
  console.log(`  Default:     ${doc.default || 'none'}`)
  // The engine's own warning when it sent one; an engine too old to send it
  // still stored the limits without enforcing them, so say it from here.
  for (const n of [...defaultNotes(doc), ...(body.warnings ?? limitNotes(doc))]) console.log(`  Note: ${n}`)
}

// The capability walker acts on one default only: defer holds the tools no rule
// names (or hands them to the Guardian Agent). A default of deny or permit is
// skipped by design (engine.js walkCapabilityRules), so a file that sets one is
// told it does nothing rather than left to believe it blocks or allows.
function defaultNotes (doc) {
  const d = doc.default
  if (!d || d === 'defer' || d === 'escalate') return []
  return [`default: ${d} has no effect. Only defer acts on the tools no rule names; otherwise the rest of the engine decides them.`]
}

// Mirrors the warning POST /brain/api/policies/import returns, for lint.
function limitNotes (doc) {
  const count = (k) => Object.keys(doc[k] || {}).length
  if (!count('rate_limits') && !count('budget')) return []
  return ["rate_limits and budget are stored with the policy but not enforced. To limit how often a tool runs, give a policy a condition that names it and add a cap in that policy's Rate limits & quotas. A spending budget has no equivalent: the engine only knows a cost the caller sends with a check, and neither the SDK, the plugins nor the gateway sends one."]
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
    const listed = toolListRefusal(r.tool)
    if (listed) die(`${listed} Rule: ${JSON.stringify(r)}`)
  }
}

// A rule names one tool. The engine reads "notes.read, notes.write" as one name,
// which no tool has, so the import route refuses such a rule; lint says so
// first, in the route's words (brain/src/lib/capability_tool.js).
function toolListRefusal (pattern) {
  if (typeof pattern !== 'string' || !pattern.includes(',')) return null
  const tools = pattern.split(',').map((t) => t.trim()).filter(Boolean)
  const plain = tools.length > 0 && tools.every((t) => !/[\s{}]/.test(t))
  const fix = !plain ? 'Write one rule per tool.'
    : tools.length === 1 ? `Write it as \`${tools[0]}\`.`
      : `Write one rule per tool: ${tools.map((t) => `\`${t}\``).join(', ')}.`
  return `A rule names one tool. \`${pattern}\` is read as a single name, which no tool has, so the rule would never apply. ${fix}`
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
  CLEVR_API_KEY   For push: a console session token, an administrator's when
                  the file permits a tool (the agent key is refused).
                  For online verify: the workspace API key.
                  Not needed for lint or offline bundle verification.

Example clevr.yaml:
  version: 1
  agent:   support-bot
  default: defer                  # tools no rule names are held
  rules:
    - permit: kb.search
    - permit: ticket.update
    - defer:  email.send          # mail outside acme.com (no @)
      if:     target.domain != "acme.com"
    - deny:   delete.*
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
