// examples/anthropic-guarded.mjs — minimal demo of @clevr/sdk + Anthropic.
//
// Shows the SDK-hook integration model: the agent uses Anthropic tools_use,
// the Clevr adapter routes every tool call through /v1/evaluate before
// running the actual tool. Block / escalate verdicts are surfaced to the
// model as tool_result errors so the model adapts (or stops).
//
// Run (after `npm install @anthropic-ai/sdk`):
//   ANTHROPIC_API_KEY=sk-...  CLEVR_API_KEY=clevr_sk_...  node examples/anthropic-guarded.mjs
//
// Without ANTHROPIC_API_KEY, this file demonstrates the wiring offline by
// stubbing the model with a fake response that returns a tool_use block.

import { Clevr } from '../index.js';
import { withClevr } from '../adapters/anthropic.js';

// --- Anthropic client (real or stubbed) ---
let anthropic;
if (process.env.ANTHROPIC_API_KEY) {
  const { default: Anthropic } = await import('@anthropic-ai/sdk');
  anthropic = new Anthropic();
} else {
  // Offline stub: returns one tool_use then a final answer.
  let calls = 0;
  anthropic = {
    messages: {
      async create({ tools }) {
        calls++;
        if (calls === 1) {
          return {
            stop_reason: 'tool_use',
            content: [
              { type: 'text', text: 'I will search the knowledge base.' },
              { type: 'tool_use', id: 'tu_1', name: 'kb_search', input: { query: 'churn risks' } },
            ],
          };
        }
        if (calls === 2) {
          return {
            stop_reason: 'tool_use',
            content: [
              { type: 'tool_use', id: 'tu_2', name: 'db_purge', input: { table: 'customers' } },
            ],
          };
        }
        return {
          stop_reason: 'end_turn',
          content: [{ type: 'text', text: 'Final answer: churn drivers identified.' }],
        };
      },
    },
  };
}

const clevr = new Clevr({
  agent: 'demo-researcher',
  onEscalate: 'throw',
});

const guarded = withClevr(anthropic, clevr);

const result = await guarded.runAgent({
  model: 'claude-sonnet-4-5',
  system: 'Research assistant. Search the kb, then summarize.',
  messages: [{ role: 'user', content: 'Find churn risks in Q1.' }],
  tools: [
    { name: 'kb_search', description: 'Search the knowledge base', input_schema: { type: 'object', properties: { query: { type: 'string' } } } },
    { name: 'db_purge', description: 'Purge a database table', input_schema: { type: 'object', properties: { table: { type: 'string' } } } },
  ],
  toolImplementations: {
    kb_search: async ({ query }) => `Found 12 docs matching "${query}"`,
    db_purge:  async ({ table }) => `[would purge ${table}]`,
  },
  toolMeta: {
    kb_search: { action_type: 'read',   target: ({ query }) => `kb/${query}` },
    db_purge:  { action_type: 'delete', target: ({ table }) => `db/${table}` },
  },
});

console.log('\n=== Session:', clevr.sessionId, '===');
console.log('Evaluations:', result.evaluations.length);
for (const e of result.evaluations) {
  console.log(`  · ${e.verdict.effect.padEnd(8)} ${e.verdict.reason}`);
}
console.log('\nFinal:', JSON.stringify(result.final.content, null, 2));
