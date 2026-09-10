// @clevr/sdk/openai — adapter for the OpenAI Chat Completions tool_calls loop.
//
// Wraps OpenAI's client.chat.completions.create({ tools, ... }) so that EVERY
// tool_call returned by the model gets routed through /v1/evaluate before the
// tool implementation actually runs. Handles the full agent loop: ask the
// model → get tool_calls → guard each → execute → feed tool results back →
// repeat until the model returns a final message with no tool_calls.
//
// Works equally well with OpenAI's official `openai` SDK, Azure OpenAI, and
// any OpenAI-compatible endpoint (Groq, Together, Mistral via OpenAI mode,
// LM Studio, etc.) — we only depend on the chat.completions shape, not on
// internal SDK types.
//
// Usage:
//   import OpenAI from 'openai';
//   import { Clevr } from '@clevr/sdk';
//   import { withClevr } from '@clevr/sdk/openai';
//
//   const openai = new OpenAI();
//   const clevr  = new Clevr({ agent: 'researcher' });
//   const guarded = withClevr(openai, clevr);
//
//   const result = await guarded.runAgent({
//     model: 'gpt-4o-mini',
//     messages: [
//       { role: 'system', content: 'You are a research assistant.' },
//       { role: 'user',   content: 'Find churn risks in Q1.' }
//     ],
//     tools: [{
//       type: 'function',
//       function: {
//         name: 'kb_search',
//         description: 'Search the knowledge base.',
//         parameters: { type: 'object', properties: { query: { type: 'string' } }, required: ['query'] }
//       }
//     }],
//     toolImplementations: {
//       kb_search: async ({ query }) => await myKb.search(query),
//     },
//     // Optional per-tool metadata for the engine to gate on
//     toolMeta: {
//       kb_search: { action_type: 'read', target: ({ query }) => `kb/${query}` },
//     },
//   });

export function withClevr(openai, clevr) {
  return {
    /**
     * Run a full OpenAI agent loop with Clevr policy gating on every tool call.
     * Returns { final, messages, evaluations }.
     */
    async runAgent({
      model, messages, tools, toolImplementations, toolMeta = {},
      maxIterations = 10, ...rest
    }) {
      const history = [...messages];
      const evaluations = [];

      // First-iteration session bootstrap — pull user's first message as goal.
      if (!clevr.sessionId) {
        const firstUser = history.find((m) => m.role === 'user');
        const goalText = typeof firstUser?.content === 'string'
          ? firstUser.content : JSON.stringify(firstUser?.content || {}).slice(0, 200);
        clevr.startSession({ goal: goalText });
      }

      for (let i = 0; i < maxIterations; i++) {
        const resp = await openai.chat.completions.create({
          model, messages: history, tools, ...rest,
        });
        const msg = resp.choices?.[0]?.message;
        if (!msg) throw new Error('OpenAI returned no message.');

        // No tool_calls → final answer, we're done.
        if (!msg.tool_calls || msg.tool_calls.length === 0) {
          history.push(msg);
          return { final: resp, messages: history, evaluations };
        }

        // The assistant message MUST be pushed before any tool message — that
        // order is part of the OpenAI tool_calls contract.
        history.push(msg);

        // Snapshot the conversation (system + prior turns + this assistant
        // tool_calls message) so the engine's content floor can scan the
        // PROMPT for PII, secrets, and prompt-injection — not just the tool
        // arguments. tool_result messages are excluded by the engine's content
        // scan, so this stays verdict-safe.
        const conversation = history.slice();

        for (const tc of msg.tool_calls) {
          const name = tc.function?.name || 'unknown_tool';
          // tc.function.arguments is a JSON string per the OpenAI contract;
          // some compatible providers send objects, accept both.
          let parsedArgs;
          try {
            parsedArgs = typeof tc.function?.arguments === 'string'
              ? JSON.parse(tc.function.arguments || '{}')
              : (tc.function?.arguments || {});
          } catch {
            parsedArgs = { _raw: tc.function?.arguments };
          }

          const meta = toolMeta[name] || {};
          const target = typeof meta.target === 'function' ? meta.target(parsedArgs) : meta.target;

          const action = {
            tool: name,
            action_type: meta.action_type || 'tool_call',
            action: `${name}(${JSON.stringify(parsedArgs).slice(0, 200)})`,
            target: target || null,
            environment: meta.environment || null,
            conversation,
            metadata: { tool_call_id: tc.id, input: parsedArgs },
          };

          let toolResult;
          const fn = toolImplementations[name];
          if (!fn) {
            // Still evaluate for the audit trail; nothing to run.
            try { await clevr.evaluate(action); } catch {}
            toolResult = `[No implementation for tool ${name}]`;
          } else {
            try {
              // ONE guard() per tool: evaluates exactly once (no double-seal),
              // respects shadow mode (always runs + records), throws on block,
              // and honors onEscalate (wait polls the SAME decision). Previously
              // this evaluate()'d then re-guard()'d on escalate (double seal) and
              // ignored shadow mode (suppressing the tool).
              const out = await clevr.guard(action, async () => fn(parsedArgs));
              const r = clevr.mode === 'shadow' ? out?.result : out;
              toolResult = typeof r === 'string' ? r : JSON.stringify(r);
            } catch (e) {
              toolResult = `[Clevr: ${e.message}]`;
            }
          }
          evaluations.push({ tool_call_id: tc.id, verdict: clevr.lastVerdict });

          history.push({
            role: 'tool',
            tool_call_id: tc.id,
            content: toolResult,
          });
        }
        // Loop back so the model can read the tool results and decide.
      }
      throw new Error(`Agent loop did not converge after ${maxIterations} iterations.`);
    },
  };
}
