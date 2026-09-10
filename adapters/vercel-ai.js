// @clevr/sdk/vercel-ai — adapter for the Vercel AI SDK (`ai` package).
//
// The Vercel AI SDK exposes `generateText` / `streamText` with a `tools`
// option where each tool is `{ description, parameters, execute }`. The
// `execute` function is what actually runs when the model calls the tool —
// so the integration point is simply to wrap `execute` with `clevr.guard`.
//
// We provide two helpers:
//   • `guardedTool(clevr, tool, meta?)` — wrap a single tool definition.
//   • `guardTools(clevr, tools, metaByName?)` — wrap an object of tools by name.
//
// Both keep the original tool shape (description, parameters) intact so the
// model sees the same schema — only the execute side becomes gated.
//
// Usage:
//   import { generateText, tool } from 'ai';
//   import { openai } from '@ai-sdk/openai';
//   import { z } from 'zod';
//   import { Clevr } from '@clevr/sdk';
//   import { guardTools } from '@clevr/sdk/vercel-ai';
//
//   const clevr = new Clevr({ agent: 'researcher' });
//
//   const tools = {
//     kb_search: tool({
//       description: 'Search the knowledge base.',
//       parameters: z.object({ query: z.string() }),
//       execute: async ({ query }) => await myKb.search(query),
//     }),
//   };
//
//   const guarded = guardTools(clevr, tools, {
//     kb_search: { action_type: 'read', target: ({ query }) => `kb/${query}` },
//   });
//
//   const result = await generateText({
//     model: openai('gpt-4o-mini'),
//     tools: guarded,
//     prompt: 'Find churn risks in Q1.',
//   });

import { ClevrBlockedError, ClevrEscalatedError } from '../index.js';

/**
 * Wrap a single Vercel AI SDK tool with a Clevr policy gate.
 *
 * @param {object} clevr  — the Clevr client
 * @param {string} name   — logical tool name surfaced in the audit log
 * @param {object} tool   — Vercel AI SDK tool definition
 *                          { description, parameters, execute }
 * @param {object} meta   — optional: { action_type, target, environment }
 *                          `target` may be a string OR a fn(args) → string
 * @returns the same tool shape, with execute replaced by a gated version.
 */
export function guardedTool(clevr, name, tool, meta = {}) {
  const original = tool.execute;
  if (typeof original !== 'function') {
    throw new Error(`Clevr vercel-ai: tool "${name}" has no execute() function.`);
  }
  return {
    ...tool,
    async execute(args, ctx) {
      const target = typeof meta.target === 'function' ? meta.target(args) : meta.target;
      // When the AI SDK passes the running messages to execute (v4+ hands
      // execute a second arg shaped { toolCallId, messages, abortSignal }),
      // forward the FULL conversation so the engine's content floor can scan
      // the PROMPT (PII / secrets / prompt-injection), not just the tool args.
      // The engine drops tool_result blocks from its content scan, so passing
      // the history stays verdict-safe. No-op on SDK versions that do not pass
      // messages, so this never breaks an older integration.
      const conversation = Array.isArray(ctx?.messages) ? ctx.messages.map(toTurn) : undefined;
      // Situate the gated tool call inside the session: when a conversation is
      // available and the caller has not already set a goal, capture the first
      // user turn as the session goal so this session reads with meaning (the
      // engine records session_goal on the first decision). A caller-provided
      // goal is respected.
      if (conversation && !clevr.sessionId && !clevr.sessionGoal) {
        const u = conversation.find((m) => m.role === 'user');
        if (u && typeof u.content === 'string') clevr.sessionGoal = u.content.slice(0, 300);
      }
      try {
        const r = await clevr.guard({
          tool: name,
          action_type: meta.action_type || 'tool_call',
          action: `${name}(${safeJson(args)})`,
          target: target || null,
          environment: meta.environment || null,
          conversation,
          metadata: { args, tool_call_id: ctx?.toolCallId },
        }, async () => original(args, ctx));
        return r;
      } catch (e) {
        // The Vercel AI SDK feeds tool errors back to the model as tool
        // results. We surface a string so the model can react gracefully.
        if (e instanceof ClevrBlockedError) {
          return `[Clevr blocked this action: ${e.verdict?.reason || e.message}]`;
        }
        if (e instanceof ClevrEscalatedError) {
          return `[Clevr requires human approval before this action runs: ${e.verdict?.reason || e.message}]`;
        }
        throw e;
      }
    },
  };
}

/**
 * Wrap an object of Vercel AI SDK tools, keyed by name.
 * `metaByName` is an optional map { kb_search: { action_type, target, ... } }.
 */
export function guardTools(clevr, tools, metaByName = {}) {
  const out = {};
  for (const [name, tool] of Object.entries(tools)) {
    out[name] = guardedTool(clevr, name, tool, metaByName[name] || {});
  }
  return out;
}

/**
 * Wrap a Vercel AI SDK generate/stream function (generateText / streamText) so
 * the PROMPT is evaluated by Clevr BEFORE the model runs — prompt-level
 * guardrails (PII / secrets / injection) in addition to the tool gate. Pair
 * with guardTools to govern BOTH the prompt and the tool calls.
 *
 *   import { generateText } from 'ai';
 *   const generate = guardGenerate(clevr, generateText);
 *   await generate({ model, messages, tools: guardTools(clevr, tools) });
 */
export function guardGenerate(clevr, generateFn, opts = {}) {
  const tool = opts.tool || 'llm.generate';
  return async (args = {}) => {
    const convo = Array.isArray(args.messages) ? args.messages.map(toTurn) : null;
    const text = convo
      ? convo.map((m) => m.content).join('\n')
      : (typeof args.prompt === 'string' ? args.prompt
        : (typeof args.system === 'string' ? args.system : ''));
    const verdict = await clevr.evaluate({
      tool,
      action_type: 'chat',
      action: text,
      conversation: convo || (text ? [{ role: 'user', content: text }] : null),
    });
    if (clevr.mode === 'enforce') {
      if (verdict.effect === 'block') throw new ClevrBlockedError(verdict);
      if ((verdict.effect === 'escalate' || verdict.effect === 'step_up') && clevr.onEscalate === 'throw') {
        throw new ClevrEscalatedError(verdict);
      }
    } else if (verdict.effect !== 'allow') {
      clevr.lastVerdict = verdict;
    }
    return await generateFn(args);
  };
}

function toTurn(m) {
  const c = m?.content;
  const content = typeof c === 'string' ? c
    : Array.isArray(c) ? c.map((p) => (typeof p === 'string' ? p : (p?.text || ''))).join('\n')
    : '';
  return { role: m?.role || 'user', content };
}

function safeJson(v) {
  try { return JSON.stringify(v).slice(0, 200); } catch { return String(v).slice(0, 200); }
}
