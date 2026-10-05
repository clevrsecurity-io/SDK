// @clevr/sdk/claude-agent-sdk — Anthropic Claude Agent SDK integration.
//
// The Claude Agent SDK (formerly Computer Use / claude-code-sdk) lets you
// build agents that drive a sandbox via a streaming agent loop. Tools are
// passed as `tools: [{ name, description, input_schema, ... }]` and tool
// implementations are provided in a `handleToolCall` callback.
//
// We wrap that callback so every tool invocation is gated by Clevr before
// the real handler runs. The tool definitions stay intact so Claude sees
// the same schema — only execution is intercepted.
//
// Usage:
//   import { query } from '@anthropic-ai/claude-agent-sdk'
//   import { Clevr } from '@clevr/sdk'
//   import { guardToolHandler } from '@clevr/sdk/claude-agent-sdk'
//
//   const clevr = new Clevr({ agent: 'coder' })
//
//   for await (const event of query({
//     prompt: 'Refactor src/foo.ts',
//     options: {
//       tools: [{ name: 'edit_file', description: '...', input_schema: { ... } }],
//       handleToolCall: guardToolHandler(clevr, async (name, input) => {
//         if (name === 'edit_file') return await editFile(input)
//         throw new Error(`Unknown tool ${name}`)
//       }, {
//         edit_file: { action_type: 'write', target: ({ path }) => path }
//       })
//     }
//   })) { ... }

import { ClevrBlockedError, ClevrEscalatedError } from '../index.js';

/**
 * Wrap a Claude Agent SDK handleToolCall function with a Clevr policy gate.
 *
 * @param {object} clevr        — the Clevr client
 * @param {function} originalHandler — async (name, input) ⇒ result
 * @param {object} toolMeta     — optional per-tool metadata for the engine
 * @returns a new handler suitable for `options.handleToolCall`
 */
export function guardToolHandler(clevr, originalHandler, toolMeta = {}) {
  return async function gatedHandler(name, input) {
    const meta = toolMeta[name] || {};
    const target = typeof meta.target === 'function' ? meta.target(input) : meta.target;
    try {
      return await clevr.guard({
        tool: name,
        runtime: clevr.runtime || 'claude-agent-sdk',   // the platform, unless the client names one
        action_type: meta.action_type || 'tool_call',
        action: `${name}(${safeJson(input)})`,
        target: target || null,
        environment: meta.environment || null,
        metadata: { input },
      }, async () => originalHandler(name, input));
    } catch (e) {
      if (e instanceof ClevrBlockedError) {
        return { error: `[Clevr blocked this action: ${e.verdict?.reason || e.message}]` };
      }
      if (e instanceof ClevrEscalatedError) {
        return { error: `[Clevr requires human approval: ${e.verdict?.reason || e.message}]` };
      }
      throw e;
    }
  };
}

function safeJson(v) {
  try { return JSON.stringify(v).slice(0, 200); } catch { return String(v).slice(0, 200); }
}
