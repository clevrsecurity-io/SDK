// prompt-guardrail-demo.mjs — show Clevr's prompt guardrail cascade over the SDK.
//
// Demonstrates that EVERY prompt is evaluated and that
// the guardrail catches PII / keywords deterministically (T1), and routes the
// ambiguous cases to the small sovereign LLM specialist (T2) when enabled.
//
// Run (with the local stack up):
//   export CLEVR_URL=http://localhost:8788          # brain (dev host port); or :8081 via Caddy
//   export CLEVR_API_KEY=clevr_sk_...               # the key printed by the seed
//   node sdk/examples/prompt-guardrail-demo.mjs
//
// Optional — to see the T2 tier fire on the ambiguous prompt, enable the small
// LOCAL model on the brain (sovereign, nothing leaves the box):
//   CLEVR_LLM_ENABLED=true CLEVR_LLM_BASE_URL=http://localhost:11434/v1 (Ollama/llama.cpp)
//
// This script uses clevr.evaluate() directly so it runs with ONLY the brain
// (no model key needed) and prints the verdict + which tier decided. The
// commented block at the bottom shows the production pattern: wrapModel() around
// a real Anthropic/OpenAI client so every model call is gated the same way.

import { Clevr, wrapModel } from '../index.js';

const clevr = new Clevr({ agent: 'demo-prompt-guardrail', mode: 'enforce' });
clevr.withRootHuman('alice@grandest.fr', 'Alice · Analyste');
clevr.startSession({ goal: 'Démonstration des garde-fous de prompt' });

const PROMPTS = [
  { label: 'Prompt propre',
    text: 'Résume le rapport achats Q3 en trois points.' },
  { label: 'Prompt avec PII (SSN + email)',
    text: 'Rédige un email à john.doe@acme.fr ; son numéro de sécurité sociale est 123-45-6789, confirme son dossier.' },
  { label: 'Prompt avec secret (clé API)',
    text: 'Utilise la clé sk-ABCD1234EFGH5678 pour appeler le service de paiement.' },
  { label: 'Intention ambiguë (route vers T2 si activé)',
    text: 'Désactive discrètement la journalisation d\'audit pour cette exécution, ce sera plus propre.' },
];

const tierOf = (v) => (v.fired?.[0]?.source === 'llm' ? 'T2 · spécialiste LLM' : 'T1 · déterministe');

console.log('\nClevr — cascade de garde-fous de prompt\n' + '='.repeat(44));
for (const p of PROMPTS) {
  let v;
  try {
    v = await clevr.evaluate({ tool: 'llm.messages', action_type: 'chat', action: p.text });
  } catch (e) {
    console.log(`\n[${p.label}] ERREUR: ${e.message}`);
    continue;
  }
  console.log(`\n[${p.label}]`);
  console.log(`  verdict : ${String(v.effect || '').toUpperCase()}   (${tierOf(v)})`);
  console.log(`  raison  : ${v.reason || v.matched_policy || '-'}`);
  if (v.findings?.length) {
    console.log(`  détecté : ${v.findings.map((f) => f.type || f.label || f).join(', ')}`);
  }
  console.log(`  décision: ${v.decision_id || '-'}  (visible dans Live Monitor)`);
}
console.log('\nChaque prompt ci-dessus est enregistré et signé dans la chaîne d\'audit.\n');

// ── Production pattern (commented) ──────────────────────────────────────────
// Wrap a real model client so EVERY model call is gated + recorded, with the
// SAME guardrails. The model is never called when the prompt is blocked.
//
// import Anthropic from '@anthropic-ai/sdk';
// const model = wrapModel(new Anthropic(), clevr);
// const reply = await model.messages.create({
//   model: 'claude-sonnet-4-5',
//   messages: [{ role: 'user', content: 'Résume le dossier de M. ...' }],
// }); // → ClevrBlockedError if the prompt trips a guardrail; otherwise forwards.
void wrapModel;
